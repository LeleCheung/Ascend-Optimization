"""Bounded diagnostic #2: tile size and native-vs-staging sweep for nan_to_num_.

Measures device-side cost only (synced walltime around a launch loop of this
candidate's own Triton kernels) and verifies each variant bit-exactly against
the validated fp32-staging kernel.  Every step is guarded so a UB-overflow
compile failure does not abort the sweep.
"""

import json
import os
import time

import torch
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "")


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
def v_native_cs(a_ptr, n, nan_v, pos_v, neg_v, BLOCK: tl.constexpr):
    dt = a_ptr.dtype.element_ty
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a_ptr + offs)
    y = tl.where(x != x, nan_v.to(dt), x)
    y = tl.where(y == float("inf"), pos_v.to(dt), y)
    y = tl.where(y == float("-inf"), neg_v.to(dt), y)
    tl.store(a_ptr + offs, y)


@triton.jit
def v_native_clamp(a_ptr, n, nan_v, pos_v, neg_v, BLOCK: tl.constexpr):
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


STREAM = get_stream()
ROWS = []


def bench(jitfn, buf, n, pos, neg, block, iters=40):
    grid = triton.cdiv(n, block)
    if jitfn is v_empty:
        sample, args = (buf, n), (buf.data_ptr(), n, block)
    else:
        sample = (buf, n, 0.0, pos, neg)
        args = (buf.data_ptr(), n, 0.0, pos, neg, block)
    ck = jitfn.warmup(*sample, BLOCK=block, grid=(1,))
    launch, fn, md = ck.run.launch, ck.function, ck.packed_metadata
    call = lambda: launch(grid, 1, 1, STREAM, fn, md, None, None, None, *args)
    for _ in range(5):
        call()
    torch.npu.synchronize()
    best = None
    for _ in range(2):
        t0 = time.perf_counter()
        for _ in range(iters):
            call()
        torch.npu.synchronize()
        dt = (time.perf_counter() - t0) / iters * 1e3
        best = dt if best is None else min(best, dt)
    return best


def edge_fill(x, gen):
    x.copy_(torch.randn(x.shape, generator=gen).to(x.dtype))
    flat = x.view(-1)
    if flat.numel() >= 7:
        flat[:7] = torch.tensor([0.0, -0.0, float("nan"), float("inf"),
                                 float("-inf"), -1.0, 1.0], dtype=x.dtype,
                                device=x.device)
    return x


def bits(t):
    return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int16)


def main():
    gen = torch.Generator().manual_seed(1729)
    cpu = torch.randn(4096 * 4096, generator=gen)
    BIG = 4096 * 4096
    gsmall = torch.randn(1024, generator=gen)

    def make(dtype, n):
        if n == BIG:
            x = cpu.to(dtype).to("npu").clone()
        else:
            x = gsmall.to(dtype).to("npu").clone()
        flat = x.view(-1)
        flat[:7] = torch.tensor([0.0, -0.0, float("nan"), float("inf"),
                                 float("-inf"), -1.0, 1.0], dtype=dtype,
                                device="npu")
        return x

    def step(tag, jitfn, dtype, n, block):
        fin = torch.finfo(dtype)
        pos, neg = fin.max, fin.min
        try:
            gold = make(dtype, n)
            gblk = 8192 if (n % 8192) == 0 else 1024
            if n % gblk:
                v_stage_masked[(triton.cdiv(n, gblk),)](gold, n, 0.0, pos, neg,
                                                        BLOCK=gblk)
            else:
                v_stage_nomask[(n // gblk,)](gold, n, 0.0, pos, neg, BLOCK=gblk)
            torch.npu.synchronize()
            buf = gold.clone()
            jitfn[(triton.cdiv(n, block),)](buf, n, 0.0, pos, neg, BLOCK=block)
            torch.npu.synchronize()
            same = bool(torch.equal(bits(buf), bits(gold)))
            ms = bench(jitfn, buf, n, pos, neg, block)
            row = {"tag": tag, "dtype": str(dtype), "n": n, "block": block,
                   "ms": ms, "bit_exact": same}
        except Exception as exc:
            row = {"tag": tag, "dtype": str(dtype), "n": n, "block": block,
                   "ms": None, "error": type(exc).__name__}
        ROWS.append(row)
        print("MEAS", json.dumps(row), flush=True)

    F16, F32, BF = torch.float16, torch.float32, torch.bfloat16
    step("f16/native_cs_b16k", v_native_cs, F16, BIG, 16384)
    step("f16/native_cs_b32k", v_native_cs, F16, BIG, 32768)
    step("f16/native_clamp_b32k", v_native_clamp, F16, BIG, 32768)
    step("f16/stage_b16k", v_stage_nomask, F16, BIG, 16384)
    step("f32/native_cs_b8k", v_native_cs, F32, BIG, 8192)
    step("f32/native_cs_b16k", v_native_cs, F32, BIG, 16384)
    step("f32/stage_b16k", v_stage_nomask, F32, BIG, 16384)
    step("bf16/native_cs_b16k", v_native_cs, BF, BIG, 16384)
    step("bf16/native_clamp_b16k", v_native_clamp, BF, BIG, 16384)
    step("f16/native_cs_b1k", v_native_cs, F16, 1024, 1024)
    try:
        buf = make(F16, 1024)
        ms = bench(v_empty, buf, 1024, 0.0, 0.0, 1024)
        ROWS.append({"tag": "f16/empty_kernel", "ms": ms})
        print("MEAS", json.dumps(ROWS[-1]), flush=True)
    except Exception as exc:
        print("MEAS empty failed", type(exc).__name__, flush=True)

    with open(os.path.join(ART, "variants2.json"), "w") as fh:
        json.dump(ROWS, fh, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
