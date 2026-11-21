"""Roofline case study for the PR #10857 lesson: place H3-shaped kernels on the B200 roofline.

Self-contained Triton kernels (not the vllm-omni ones) so the numbers illustrate the principles only.
Rows default to 32768 tokens, hidden 5376 (H3). Prints one line per kernel: time, model DRAM bytes,
FLOPs, intensity, achieved GB/s and GFLOP/s.
"""

import os
import statistics

import torch
import triton
import triton.language as tl

ROWS = int(os.environ.get("ROWS", 32768))
HIDDEN = 5376
TABLE_ROWS = 12
EPS = 1e-6
# NCU_MODE=1: one warm-up launch, then exactly one launch inside an NVTX range named after the case, so that
# Nsight Compute with --nvtx-include "<case>/" captures each typical case once.
NCU_MODE = bool(os.environ.get("NCU_MODE"))


@triton.jit
def gate_row_kernel(out, x, g, b, idx, n, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < n
    k = tl.load(idx + row)
    xv = tl.load(x + row * n + cols, mask=m, other=0.0).to(tl.float32)
    gv = tl.load(g + k * n + cols, mask=m, other=0.0).to(tl.float32)
    bv = tl.load(b + row * n + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(out + row * n + cols, xv + gv * bv, mask=m)


@triton.jit
def gate_col_kernel(out, x, g, b, idx, n, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < n
    k = tl.load(idx + row)
    xv = tl.load(x + row * n + cols, mask=m, other=0.0).to(tl.float32)
    gv = tl.load(g + k * n + cols, mask=m, other=0.0).to(tl.float32)
    bv = tl.load(b + row * n + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(out + row * n + cols, xv + gv * bv, mask=m)


@triton.jit
def norm_mod_kernel(out, y, w, shift, scale, idx, n, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < n
    k = tl.load(idx + row)
    yv = tl.load(y + row * n + cols, mask=m, other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(yv * yv, axis=0) / n + eps)
    wv = tl.load(w + cols, mask=m, other=0.0).to(tl.float32)
    normed = (yv * inv * wv).to(tl.bfloat16).to(tl.float32)
    sh = tl.load(shift + k * n + cols, mask=m, other=0.0).to(tl.float32)
    sc = tl.load(scale + k * n + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(out + row * n + cols, normed * (1.0 + sc) + sh, mask=m)


@triton.jit
def fused_gate_norm_mod_kernel(y_out, out, x, g, b, w, shift, scale, idx, n, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    m = cols < n
    k = tl.load(idx + row)
    xv = tl.load(x + row * n + cols, mask=m, other=0.0).to(tl.float32)
    gv = tl.load(g + k * n + cols, mask=m, other=0.0).to(tl.float32)
    bv = tl.load(b + row * n + cols, mask=m, other=0.0).to(tl.float32)
    yv = (xv + gv * bv).to(tl.bfloat16)
    tl.store(y_out + row * n + cols, yv, mask=m)
    yf = yv.to(tl.float32)
    inv = tl.rsqrt(tl.sum(yf * yf, axis=0) / n + eps)
    wv = tl.load(w + cols, mask=m, other=0.0).to(tl.float32)
    normed = (yf * inv * wv).to(tl.bfloat16).to(tl.float32)
    sh = tl.load(shift + k * n + cols, mask=m, other=0.0).to(tl.float32)
    sc = tl.load(scale + k * n + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(out + row * n + cols, normed * (1.0 + sc) + sh, mask=m)


def bench(fn, warmup=20, iters=100, name=None):
    if NCU_MODE:
        fn()
        torch.cuda.synchronize()
        if name:
            torch.cuda.nvtx.range_push(name)
        fn()
        torch.cuda.synchronize()
        if name:
            torch.cuda.nvtx.range_pop()
        return float("nan")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1e-3)
    return statistics.median(times)


def report(name, seconds, dram_bytes, flops):
    gbs = dram_bytes / seconds / 1e9
    gflops = flops / seconds / 1e9
    intensity = flops / dram_bytes
    print(f"ROW|{name}|{seconds * 1e6:.1f}|{dram_bytes / 1e6:.1f}|{flops / 1e9:.3f}|{intensity:.3f}|{gbs:.0f}|{gflops:.1f}", flush=True)


def decode_sweep(dev):
    """Decode-like GEMM y = x @ W: M token vectors against a fixed weight that does not fit in L2.

    Bytes = 2 (MK + KN + MN), FLOPs = 2 MKN, so intensity is about M for small M: the same operation moves
    from memory-bound to compute-bound as the batch grows. Each M runs inside an NVTX range decode_M<M>.
    """
    k, n = 8192, 28672
    w = torch.randn(k, n, device=dev, dtype=torch.bfloat16)
    ms = [int(m) for m in (os.environ.get("DECODE_MS") or "1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192").split()]
    for m in ms:
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        t = bench(lambda: torch.mm(x, w), warmup=10, iters=50, name=f"decode_M{m}")
        report(f"decode GEMM M={m} K={k} N={n}", t, 2 * (m * k + k * n + m * n), 2 * m * k * n)


def main():
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    print(f"GPU {props.name} SMs={props.multi_processor_count} L2={props.L2_cache_size / 2**20:.0f}MiB rows={ROWS} hidden={HIDDEN}", flush=True)
    torch.manual_seed(0)
    if os.environ.get("PART") == "decode":  # PART=decode: run only the decode sweep
        decode_sweep(dev)
        return
    bf = dict(device=dev, dtype=torch.bfloat16)
    x = torch.randn(ROWS, HIDDEN, **bf)
    b = torch.randn(ROWS, HIDDEN, **bf)
    g = 0.3 * torch.randn(TABLE_ROWS, HIDDEN, **bf)
    shift = 0.3 * torch.randn(TABLE_ROWS, HIDDEN, **bf)
    scale = 0.3 * torch.randn(TABLE_ROWS, HIDDEN, **bf)
    w = (1 + 0.1 * torch.randn(HIDDEN, device=dev)).to(torch.bfloat16)
    idx = torch.randint(0, TABLE_ROWS, (ROWS,), device=dev)
    y_row, y_col, y_f, out_u, out_f = (torch.empty_like(x) for _ in range(5))
    elems = ROWS * HIDDEN

    # Achievable bandwidth: a plain device-to-device copy reads and writes every byte once.
    dst = torch.empty_like(x)
    t = bench(lambda: dst.copy_(x))
    report("copy (bandwidth probe)", t, 2 * x.numel() * 2, 0)

    full = triton.next_power_of_2(HIDDEN)
    run_row = lambda: gate_row_kernel[(ROWS,)](y_row, x, g, b, idx, HIDDEN, BLOCK=full, num_warps=8)
    run_col = lambda: gate_col_kernel[(ROWS, triton.cdiv(HIDDEN, 2048))](y_col, x, g, b, idx, HIDDEN, BLOCK=2048, num_warps=4)
    # Gate: read x, b; write y (gate table stays in L2). 2 FLOP/element.
    report("gate, one 8192 tile per row", bench(run_row, name="gate_row"), 3 * elems * 2, 2 * elems)
    report("gate, 2048-column tiles", bench(run_col, name="gate_col"), 3 * elems * 2, 2 * elems)
    torch.cuda.synchronize()
    print(f"CHECK gate row==col: {torch.equal(y_row, y_col)}", flush=True)

    run_norm = lambda: norm_mod_kernel[(ROWS,)](out_u, y_row, w, shift, scale, idx, HIDDEN, EPS, BLOCK=full, num_warps=8)
    run_fused = lambda: fused_gate_norm_mod_kernel[(ROWS,)](y_f, out_f, x, g, b, w, shift, scale, idx, HIDDEN, EPS, BLOCK=full, num_warps=8)

    def run_unfused():
        run_row()
        run_norm()

    # Norm+modulate alone: read y, write out. ~6 FLOP/element (square, sum, scale, weight, scale, shift).
    report("norm+modulate alone", bench(run_norm, name="norm_mod"), 2 * elems * 2, 6 * elems)
    # Unfused pair: gate (6 B/elem) then norm+modulate (4 B/elem) = 10 B/elem, 8 FLOP/elem.
    report("gate then norm+modulate (2 kernels)", bench(run_unfused), 5 * elems * 2, 8 * elems)
    # Fused: read x, b; write y, out = 8 B/elem, same 8 FLOP/elem.
    report("fused gate+norm+modulate (1 kernel)", bench(run_fused, name="fused"), 4 * elems * 2, 8 * elems)
    run_unfused()
    run_fused()
    torch.cuda.synchronize()
    print(f"CHECK fused==unfused: y {torch.equal(y_f, y_row)} out {torch.equal(out_f, out_u)}", flush=True)

    # Compute-bound contrast: square BF16 GEMM.
    n = 8192
    a_m, b_m = torch.randn(n, n, **bf), torch.randn(n, n, **bf)
    report(f"GEMM {n}x{n}x{n}", bench(lambda: torch.mm(a_m, b_m), warmup=5, iters=30, name="gemm_8192"), 3 * n * n * 2, 2 * n**3)
    del a_m, b_m, x, b, y_row, y_col, y_f, out_u, out_f, dst
    decode_sweep(dev)


if __name__ == "__main__":
    main()
