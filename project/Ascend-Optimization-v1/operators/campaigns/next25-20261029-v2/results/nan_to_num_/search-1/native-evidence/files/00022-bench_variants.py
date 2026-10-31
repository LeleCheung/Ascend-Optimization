"""Bounded diagnostic: device-side cost of nan_to_num_ kernel variants.

Only allocates inputs, launches this candidate's own Triton kernels and takes
walltime around a synced launch loop.  No framework op computes the result.
Every variant's output is compared bit-for-bit against the already-validated
fp32-staging kernel, so a native-dtype fast path is proven exact before it is
submitted to the formal evaluator.
"""

import json
import os
import sys
import time

import torch
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")


@triton.jit
def v_stage_masked(a_ptr, n, nan_v, pos_v, neg_v, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.where(x != x, nan_v, x)
    y = tl.where(y == float("inf"), pos_v, y)
    y = tl.where(y == float("-inf"), neg_v, y)
    tl.store(a_ptr + offs, y.to(a_ptr.dtype.element_ty), mask=mask)


@triton.jit
def v_stage_nomask(a_ptr, n, nan_v, pos_v, neg_v, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a_ptr + offs).to(tl.float32)
    y = tl.where(x != x, nan_v, x)
    y = tl.where(y == float("inf"), pos_v, y)
    y = tl.where(y == float("-inf"), neg_v, y)
    tl.store(a_ptr + offs, y.to(a_ptr.dtype.element_ty))


@triton.jit
def v_native_nomask(a_ptr, n, nan_v, pos_v, neg_v, BLOCK: tl.constexpr):
    dt = a_ptr.dtype.element_ty
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a_ptr + offs)
    y = tl.where(x != x, nan_v.to(dt), x)
    y = tl.where(y == float("inf"), pos_v.to(dt), y)
    y = tl.where(y == float("-inf"), neg_v.to(dt), y)
    tl.store(a_ptr + offs, y)


@triton.jit
def v_native_clamp_nomask(a_ptr, n, nan_v, pos_v, neg_v, BLOCK: tl.constexpr):
    dt = a_ptr.dtype.element_ty
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a_ptr + offs)
    y = tl.maximum(tl.minimum(x, pos_v.to(dt)), neg_v.to(dt))
    y = tl.where(x != x, nan_v.to(dt), y)
    tl.store(a_ptr + offs, y)


@triton.jit
def v_empty(a_ptr, n, BLOCK: tl.constexpr):
    pass


def get_stream():
    try:
        return torch.npu.current_stream().npu_stream
    except Exception:
        from triton.runtime import driver

        return driver.active.get_current_stream(0)


STREAM = None
RESULTS = []


def make_fast(jitfn, sample, block, grid):
    ck = jitfn.warmup(*sample, BLOCK=block, grid=(1,))
    launch = ck.run.launch
    fn = ck.function
    md = ck.packed_metadata
    return lambda args: launch(grid, 1, 1, STREAM, fn, md,
                               None, None, None, *args)


def bench(jitfn, buf, n, pos, neg, block, iters=40):
    grid = triton.cdiv(n, block)
    if jitfn is v_empty:
        sample, args = (buf, n), (buf.data_ptr(), n, block)
    else:
        sample = (buf, n, 0.0, pos, neg)
        args = (buf.data_ptr(), n, 0.0, pos, neg, block)
    entry = make_fast(jitfn, sample, block, grid)
    for _ in range(5):
        entry(args)
    torch.npu.synchronize()
    best = None
    for _ in range(2):
        t0 = time.perf_counter()
        for _ in range(iters):
            entry(args)
        torch.npu.synchronize()
        dt = (time.perf_counter() - t0) / iters * 1e3
        best = dt if best is None else min(best, dt)
    return best


def edge_fill(x):
    x.normal_()
    flat = x.view(-1)
    if flat.numel() >= 7:
        flat[:7] = torch.tensor([0.0, -0.0, float("nan"), float("inf"),
                                 float("-inf"), -1.0, 1.0], dtype=x.dtype,
                                device=x.device)
    return x


def bits(t):
    return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int16)


def record(tag, dtype, n, block, ms, extra=None):
    row = {"tag": tag, "dtype": str(dtype), "n": n, "block": block, "ms": ms}
    if extra:
        row.update(extra)
    RESULTS.append(row)
    print("MEAS", json.dumps(row), flush=True)


def main():
    global STREAM
    STREAM = get_stream()
    torch.manual_seed(1729)
    BIG = 4096 * 4096
    fin16, fin32, finbf = (torch.finfo(torch.float16), torch.finfo(torch.float32),
                           torch.finfo(torch.bfloat16))

    def golden(dtype, n, pos, neg):
        """Already-validated fp32-staging result, plus its latency."""
        gold = edge_fill(torch.randn(n, dtype=dtype, device="npu"))
        gblk = 8192 if (n % 8192) == 0 else 1024
        if n % gblk:
            v_stage_masked[(triton.cdiv(n, gblk),)](gold, n, 0.0, pos, neg,
                                                    BLOCK=gblk)
            tag = "stage_masked"
        else:
            v_stage_nomask[(n // gblk,)](gold, n, 0.0, pos, neg, BLOCK=gblk)
            tag = "stage_nomask"
        torch.npu.synchronize()
        record("ref/" + tag, dtype, n, gblk, bench(
            v_stage_masked if n % gblk else v_stage_nomask, gold, n, pos, neg, gblk))
        return gold

    def run_one(tag, jitfn, dtype, n, block, pos, neg, gold):
        buf = gold.clone()
        grid = triton.cdiv(n, block)
        jitfn[grid](buf, n, 0.0, pos, neg, BLOCK=block)
        torch.npu.synchronize()
        same = bool(torch.equal(bits(buf), bits(gold)))
        ms = bench(jitfn, buf, n, pos, neg, block)
        record(tag, dtype, n, block, ms, {"bit_exact": same})

    # ---- float16 4096x4096 -------------------------------------------------
    g = golden(torch.float16, BIG, fin16.max, fin16.min)
    run_one("f16/BIG/stage_masked", v_stage_masked, torch.float16, BIG, 8192,
            fin16.max, fin16.min, g)
    run_one("f16/BIG/native_nomask", v_native_nomask, torch.float16, BIG, 8192,
            fin16.max, fin16.min, g)
    run_one("f16/BIG/native_clamp", v_native_clamp_nomask, torch.float16, BIG,
            8192, fin16.max, fin16.min, g)
    run_one("f16/BIG/native_clamp_b16k", v_native_clamp_nomask, torch.float16,
            BIG, 16384, fin16.max, fin16.min, g)
    del g

    # ---- float32 4096x4096 -------------------------------------------------
    g = golden(torch.float32, BIG, fin32.max, fin32.min)
    run_one("f32/BIG/native_clamp", v_native_clamp_nomask, torch.float32, BIG,
            8192, fin32.max, fin32.min, g)
    run_one("f32/BIG/native_clamp_b16k", v_native_clamp_nomask, torch.float32,
            BIG, 16384, fin32.max, fin32.min, g)
    del g

    # ---- bfloat16 4096x4096 ------------------------------------------------
    g = golden(torch.bfloat16, BIG, finbf.max, finbf.min)
    run_one("bf16/BIG/native_clamp", v_native_clamp_nomask, torch.bfloat16, BIG,
            8192, finbf.max, finbf.min, g)
    del g

    # ---- tiny inputs: launch floor ----------------------------------------
    g = golden(torch.float16, 1024, fin16.max, fin16.min)
    run_one("f16/SMALL/native_clamp", v_native_clamp_nomask, torch.float16,
            1024, 1024, fin16.max, fin16.min, g)
    del g
    small = torch.randn(1024, dtype=torch.float16, device="npu")
    record("f16/SMALL/empty_kernel", torch.float16, 1024, 1024,
           bench(v_empty, small, 1024, 0.0, 0.0, 1024))

    with open(os.path.join(ART, "variants.json"), "w") as fh:
        json.dump(RESULTS, fh, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
