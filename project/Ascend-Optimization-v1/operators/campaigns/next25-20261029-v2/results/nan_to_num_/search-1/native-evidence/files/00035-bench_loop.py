"""Bounded diagnostic #3: does looping several tiles inside one program help?

Per-program fixed cost is the open question for the 4096x4096 workloads.  Each
variant below keeps the tile at a UB-safe size and only changes how many tiles a
program walks (grid shrinks accordingly), so the measurement isolates
per-program overhead / pipeline overlap from tile size.  Device-side only, and
each variant is checked bit-exactly against the validated staged kernel.
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
def v_loop_clamp(a_ptr, n, nan_v, pos_v, neg_v, BLOCK: tl.constexpr,
                 ITERS: tl.constexpr):
    dt = a_ptr.dtype.element_ty
    pid = tl.program_id(0)
    base = pid * BLOCK + tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    nan_c = nan_v.to(dt)
    pos_c = pos_v.to(dt)
    neg_c = neg_v.to(dt)
    for i in range(ITERS):
        offs = base + i * step
        mask = offs < n
        x = tl.load(a_ptr + offs, mask=mask, other=0.0)
        y = tl.maximum(tl.minimum(x, pos_c), neg_c)
        y = tl.where(x != x, nan_c, y)
        tl.store(a_ptr + offs, y, mask=mask)


@triton.jit
def v_loop_cs(a_ptr, n, nan_v, pos_v, neg_v, BLOCK: tl.constexpr,
              ITERS: tl.constexpr):
    dt = a_ptr.dtype.element_ty
    pid = tl.program_id(0)
    base = pid * BLOCK + tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    nan_c = nan_v.to(dt)
    pos_c = pos_v.to(dt)
    neg_c = neg_v.to(dt)
    for i in range(ITERS):
        offs = base + i * step
        mask = offs < n
        x = tl.load(a_ptr + offs, mask=mask, other=0.0)
        y = tl.where(x != x, nan_c, x)
        y = tl.where(y == float("inf"), pos_c, y)
        y = tl.where(y == float("-inf"), neg_c, y)
        tl.store(a_ptr + offs, y, mask=mask)


def get_stream():
    try:
        return torch.npu.current_stream().npu_stream
    except Exception:
        from triton.runtime import driver

        return driver.active.get_current_stream(0)


STREAM = get_stream()
ROWS = []


def bench(jitfn, buf, n, pos, neg, block, iters_k, iters=40):
    grid = triton.cdiv(n, block * iters_k)
    sample = (buf, n, 0.0, pos, neg)
    args = (buf.data_ptr(), n, 0.0, pos, neg, block, iters_k)
    ck = jitfn.warmup(*sample, BLOCK=block, ITERS=iters_k, grid=(1,))
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


def bits(t):
    return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int16)


def main():
    gen = torch.Generator().manual_seed(1729)
    BIG = 4096 * 4096
    cpu = torch.randn(BIG, generator=gen)

    def make(dtype):
        x = cpu.to(dtype).to("npu").clone()
        x.view(-1)[:7] = torch.tensor([0.0, -0.0, float("nan"), float("inf"),
                                       float("-inf"), -1.0, 1.0], dtype=dtype,
                                      device="npu")
        return x

    def step(tag, jitfn, dtype, block, iters_k):
        fin = torch.finfo(dtype)
        pos, neg = fin.max, fin.min
        try:
            gold = make(dtype)
            v_stage_masked[(triton.cdiv(BIG, 8192),)](gold, BIG, 0.0, pos, neg,
                                                      BLOCK=8192)
            torch.npu.synchronize()
            buf = gold.clone()
            grid = (triton.cdiv(BIG, block * iters_k),)
            jitfn[grid](buf, BIG, 0.0, pos, neg, BLOCK=block, ITERS=iters_k)
            torch.npu.synchronize()
            same = bool(torch.equal(bits(buf), bits(gold)))
            ms = bench(jitfn, buf, BIG, pos, neg, block, iters_k)
            row = {"tag": tag, "dtype": str(dtype), "block": block,
                   "iters": iters_k, "grid": grid[0], "ms": ms,
                   "bit_exact": same}
        except Exception as exc:
            row = {"tag": tag, "dtype": str(dtype), "block": block,
                   "iters": iters_k, "ms": None, "err": str(exc)[:200]}
        ROWS.append(row)
        print("MEAS", json.dumps(row), flush=True)

    F16, F32, BF = torch.float16, torch.float32, torch.bfloat16
    for k in (2, 4, 8, 16):
        step("f16/loop_clamp_b16k_i%d" % k, v_loop_clamp, F16, 16384, k)
    step("bf16/loop_clamp_b8k_i1", v_loop_clamp, BF, 8192, 1)
    step("bf16/loop_clamp_b8k_i4", v_loop_clamp, BF, 8192, 4)
    step("bf16/loop_clamp_b16k_i4", v_loop_clamp, BF, 16384, 4)
    step("f32/loop_cs_b8k_i1", v_loop_cs, F32, 8192, 1)
    step("f32/loop_cs_b8k_i4", v_loop_cs, F32, 8192, 4)
    step("f32/loop_cs_b8k_i8", v_loop_cs, F32, 8192, 8)

    with open(os.path.join(ART, "loop.json"), "w") as fh:
        json.dump(ROWS, fh, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
