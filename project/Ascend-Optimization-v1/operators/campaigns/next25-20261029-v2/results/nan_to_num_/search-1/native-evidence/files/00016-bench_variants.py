"""Bounded diagnostic: device-side cost of nan_to_num_ kernel variants.

Only allocates inputs, launches this candidate's own Triton kernels and takes
walltime around a synced launch loop.  No framework op computes the result.
Each variant's output is compared bit-for-bit against the already-validated
fp32-staging kernel so that a native-dtype fast path is proven exact before it
is submitted to the formal evaluator.
"""

import json
import os
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


def make_fast(jitfn, sample, block, grid):
    """Resolve the compiled launcher once (mirrors the candidate fast path)."""
    ck = jitfn.warmup(*sample, BLOCK=block, grid=(1,))
    launch = ck.run.launch
    fn = ck.function
    md = ck.packed_metadata
    return lambda args: launch(grid, 1, 1, STREAM, fn, md,
                               None, None, None, *args)


def bench(jitfn, buf, n, pos, neg, block, iters=60):
    grid = (triton.cdiv(n, block),)
    sample = (buf, n, 0.0, pos, neg) if jitfn is not v_empty else (buf, n)
    entry = make_fast(jitfn, sample, block, grid)
    if jitfn is v_empty:
        args = (buf.data_ptr(), n, block)
    else:
        args = (buf.data_ptr(), n, 0.0, pos, neg, block)
    for _ in range(10):
        entry(args)
    torch.npu.synchronize()
    best = None
    for _ in range(3):
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
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    return t.view(torch.int16)


def main():
    global STREAM
    STREAM = get_stream()
    torch.manual_seed(1729)
    results = []
    checks = []

    def run_one(tag, jitfn, dtype, n_elems, block):
        fin = torch.finfo(dtype)
        pos, neg = fin.max, fin.min
        src = edge_fill(torch.randn(n_elems, dtype=dtype, device="npu"))
        buf = src.clone()
        # golden: already-validated fp32-staging kernel
        gold = src.clone()
        gblk = 8192 if (n_elems % 8192) == 0 else 1024
        if n_elems % gblk:
            gfn = v_stage_masked
            ggrid = (triton.cdiv(n_elems, gblk),)
            v_stage_masked[ggrid](gold, n_elems, 0.0, pos, neg, BLOCK=gblk)
        else:
            v_stage_nomask[(n_elems // gblk,)](gold, n_elems, 0.0, pos, neg,
                                               BLOCK=gblk)
        torch.npu.synchronize()

        if jitfn is v_empty:
            ms = bench(jitfn, buf, n_elems, pos, neg, block)
            results.append({"tag": tag, "dtype": str(dtype), "n": n_elems,
                            "block": block, "ms": ms})
            return

        grid = (triton.cdiv(n_elems, block),)
        if n_elems % block:
            gfn_grid = grid
            jitfn[gfn_grid](buf, n_elems, 0.0, pos, neg, BLOCK=block)
        else:
            jitfn[grid](buf, n_elems, 0.0, pos, neg, BLOCK=block)
        torch.npu.synchronize()
        same = torch.equal(bits(buf), bits(gold))
        nan_seen = int(torch.isnan(src).sum().item())
        inf_seen = int(torch.isinf(src).sum().item())
        checks.append({"tag": tag, "dtype": str(dtype), "n": n_elems,
                       "block": block, "bit_exact_vs_staging": bool(same),
                       "nan_in": nan_seen, "inf_in": inf_seen})
        ms = bench(jitfn, buf, n_elems, pos, neg, block)
        results.append({"tag": tag, "dtype": str(dtype), "n": n_elems,
                        "block": block, "ms": ms})

    BIG = 4096 * 4096
    variants = [
        ("stage_masked", v_stage_masked),
        ("stage_nomask", v_stage_nomask),
        ("native_nomask", v_native_nomask),
        ("native_clamp_nomask", v_native_clamp_nomask),
    ]
    for name, fn in variants:
        run_one("f16_big/" + name, fn, torch.float16, BIG, 8192)
    for name in ("native_nomask", "native_clamp_nomask"):
        fn = dict(variants)[name]
        for blk in (16384, 32768, 65536):
            run_one("f16_big/%s_b%d" % (name, blk), fn, torch.float16, BIG, blk)
    for name in ("stage_masked", "native_nomask", "native_clamp_nomask"):
        run_one("f32_big/" + name, dict(variants)[name], torch.float32, BIG, 8192)
    for name in ("stage_masked", "native_nomask", "native_clamp_nomask"):
        run_one("bf16_big/" + name, dict(variants)[name], torch.bfloat16, BIG, 8192)
    run_one("f32_big/native_clamp_b16384", v_native_clamp_nomask,
            torch.float32, BIG, 16384)
    run_one("f16_small/stage_masked", v_stage_masked, torch.float16, 1024, 1024)
    run_one("f16_small/native_clamp", v_native_clamp_nomask, torch.float16, 1024, 1024)
    run_one("f16_small/empty", v_empty, torch.float16, 1024, 1024)

    out = {"results": results, "checks": checks}
    with open(os.path.join(ART, "variants.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
