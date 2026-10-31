"""Second bounded diagnosis: minimum achievable host path, and whether
input-dtype (f16/bf16) math or loop pipelining lifts big-input throughput.

Writes one compact JSON artifact. No candidate source is modified.
"""
import inspect
import json
import os
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = {"host": {}, "big": {}, "launcher": {}}


def bench(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


@triton.jit
def k_f32math(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for start in range(tl.program_id(0) * BLOCK, N, step):
        off = start + lane
        x = tl.load(X + off, off < N, other=0).to(tl.float32)
        inner = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
        tl.store(Y + off, x * inner * (1.0 / 6.0), off < N)


@triton.jit
def k_native(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for start in range(tl.program_id(0) * BLOCK, N, step):
        off = start + lane
        x = tl.load(X + off, off < N, other=0)
        inner = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
        tl.store(Y + off, x * inner * (1.0 / 6.0), off < N)


@triton.jit
def k_stage2(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, STAGES: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for start in tl.range(tl.program_id(0) * BLOCK, N, step, num_stages=STAGES):
        off = start + lane
        x = tl.load(X + off, off < N, other=0).to(tl.float32)
        inner = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
        tl.store(Y + off, x * inner * (1.0 / 6.0), off < N)


@triton.jit
def k_meta_only(X, Y):
    off = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    x = tl.load(X + off).to(tl.float32)
    tl.store(Y + off, tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) * x * 0.16666667)


def main():
    dev = torch.npu.current_device()
    raw = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
    x = torch.randn(1024, device="npu", dtype=torch.float16)
    out = torch.empty_like(x)
    n = x.numel()

    ck = k_f32math.warmup(x, out, n, 1024, grid=(1,))
    ck_meta = k_meta_only.warmup(x, out, grid=(1,))
    L = ck.run.launch
    F, P = ck.function, ck.packed_metadata
    Lm = ck_meta.run.launch
    Fm, Pm = ck_meta.function, ck_meta.packed_metadata
    stream = lambda: raw(dev)

    lau = OUT["launcher"]
    lau["type"] = str(type(ck.run))
    lau["ck_type"] = str(type(ck))
    lau["attrs"] = [a for a in dir(ck.run) if not a.startswith("_")]
    lau["ck_attrs"] = [a for a in dir(ck) if not a.startswith("_")]
    try:
        lau["launch_src"] = inspect.getsource(type(L))[:4000]
    except Exception as exc:
        lau["launch_src_error"] = repr(exc)
    try:
        lau["launch_sig"] = str(inspect.signature(L))
    except Exception as exc:
        lau["launch_sig_error"] = repr(exc)
    try:
        lau["function_type"] = str(type(F))
        lau["packed_type"] = str(type(P))
    except Exception as exc:
        lau["function_error"] = repr(exc)

    xp, yp = x.data_ptr(), out.data_ptr()
    h = OUT["host"]
    h["launch_only_13args_us"] = bench(
        lambda: L(1, 1, 1, stream(), F, P, None, None, None, xp, yp, n, 1024), 2000, 200)
    h["launch_only_2args_us"] = bench(
        lambda: Lm(1, 1, 1, stream(), Fm, Pm, None, None, None, xp, yp), 2000, 200)
    h["empty_like_us"] = bench(lambda: torch.empty_like(x), 2000, 200)
    h["new_empty_us"] = bench(lambda: x.new_empty(x.shape), 2000, 200)
    h["torch_empty_us"] = bench(
        lambda: torch.empty(x.shape, dtype=x.dtype, device=x.device), 2000, 200)
    h["ref_us"] = bench(lambda: torch.nn.functional.hardswish(x), 2000, 200)

    def combo_meta():
        o = x.new_empty(x.shape)
        Lm(1, 1, 1, stream(), Fm, Pm, None, None, None, x.data_ptr(), o.data_ptr())
        return o

    def combo_full():
        o = x.new_empty(x.shape)
        L(1, 1, 1, stream(), F, P, None, None, None,
          x.data_ptr(), o.data_ptr(), x.numel(), 1024)
        return o

    h["combo_meta_us"] = bench(combo_meta, 2000, 200)
    h["combo_full_us"] = bench(combo_full, 2000, 200)
    got = combo_meta()
    exp = torch.nn.functional.hardswish(x)
    h["combo_meta_max_abs_err"] = float((got - exp).abs().max().item())

    # ------------------------------------------------------------- big cases
    def big(tag, dt, blocks, stages):
        xx = torch.randn(4096, 4096, device="npu", dtype=dt)
        yy = torch.empty_like(xx)
        nb = xx.numel()
        expected = torch.nn.functional.hardswish(xx)
        r = {"ref_us": bench(lambda: torch.nn.functional.hardswish(xx), 60, 20)}
        for name, kern, blk in blocks:
            grid = 40
            try:
                kern.warmup(xx, yy, nb, blk, grid=(1,))
                yy.zero_()
                kern[(grid,)](xx, yy, nb, blk)
                torch.npu.synchronize()
                r[f"{name}_blk{blk}_g40_err"] = float((yy - expected).abs().max().item())
                r[f"{name}_blk{blk}_g40_us"] = bench(
                    lambda: kern[(grid,)](xx, yy, nb, blk), 60, 20)
            except Exception as exc:
                r[f"{name}_blk{blk}_g40_error"] = repr(exc)[:200]
        for blk, st in stages:
            try:
                k_stage2.warmup(xx, yy, nb, blk, st, grid=(1,))
                yy.zero_()
                k_stage2[(40,)](xx, yy, nb, blk, st)
                torch.npu.synchronize()
                r[f"stage{st}_blk{blk}_g40_err"] = float((yy - expected).abs().max().item())
                r[f"stage{st}_blk{blk}_g40_us"] = bench(
                    lambda: k_stage2[(40,)](xx, yy, nb, blk, st), 60, 20)
            except Exception as exc:
                r[f"stage{st}_blk{blk}_g40_error"] = repr(exc)[:200]
        OUT["big"][tag] = r

    big("f16", torch.float16,
        [("f32math", k_f32math, 4096), ("native", k_native, 4096),
         ("native", k_native, 8192), ("native", k_native, 16384)],
        [(4096, 2), (2048, 3)])
    big("bf16", torch.bfloat16,
        [("f32math", k_f32math, 4096), ("native", k_native, 8192)],
        [])
    big("f32", torch.float32,
        [("f32math", k_f32math, 4096), ("native", k_native, 4096),
         ("f32math", k_f32math, 8192)],
        [(2048, 3)])

    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "probe2.json"), "w") as fh:
        json.dump(OUT, fh, indent=2, sort_keys=True)
    print("WROTE probe2.json")


if __name__ == "__main__":
    main()
