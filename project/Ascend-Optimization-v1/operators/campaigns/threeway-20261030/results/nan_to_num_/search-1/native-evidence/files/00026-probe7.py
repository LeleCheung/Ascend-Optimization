"""Probe #7: sequential-loop variants of the *staged* structure for bf16/fp32,
and masked vs unmasked loop bodies (mask presence changed codegen in rounds 2-3).
"""

import json
import os
import sys
import time

sys.path.insert(0, "tmp")

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

import main as cand

DEV = torch.device("npu:0")
ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
LIMITS = cand._DEFAULT_LIMITS
STG = lambda: torch_npu._C._npu_getCurrentRawStreamNoWait(0)


@triton.jit
def _stage_loop_m(a_ptr, n, nan_v, pos_v, neg_v, ITERS: tl.constexpr,
                  BLOCK: tl.constexpr):
    base = tl.program_id(0) * (ITERS * BLOCK)
    for i in range(ITERS):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.where(x != x, nan_v, x)
        y = tl.where(x == float("inf"), pos_v, y)
        y = tl.where(x == float("-inf"), neg_v, y)
        tl.store(a_ptr + offs, y.to(a_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _clamp_loop(a_ptr, n, nan_v, pos_v, neg_v, ITERS: tl.constexpr,
                BLOCK: tl.constexpr):
    dt = a_ptr.dtype.element_ty
    base = tl.program_id(0) * (ITERS * BLOCK)
    for i in range(ITERS):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(a_ptr + offs)
        y = tl.maximum(tl.minimum(x, pos_v.to(dt)), neg_v.to(dt))
        y = tl.where(x != x, nan_v.to(dt), y)
        tl.store(a_ptr + offs, y)


@triton.jit
def _clamp_loop_m(a_ptr, n, nan_v, pos_v, neg_v, ITERS: tl.constexpr,
                  BLOCK: tl.constexpr):
    dt = a_ptr.dtype.element_ty
    base = tl.program_id(0) * (ITERS * BLOCK)
    for i in range(ITERS):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(a_ptr + offs, mask=mask, other=0.0)
        y = tl.maximum(tl.minimum(x, pos_v.to(dt)), neg_v.to(dt))
        y = tl.where(x != x, nan_v.to(dt), y)
        tl.store(a_ptr + offs, y, mask=mask)


def safe(fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "ub overflow" in msg:
            return {"error": "ub_overflow"}
        return {"error": type(exc).__name__ + ": " + msg[:150]}


def variant(jitfn, args_t, args_p, tail, grid, consts, iters=80):
    ck = jitfn.warmup(*args_t, grid=(1,), **consts)
    launcher, fn, md = ck.run.launch, ck.function, ck.packed_metadata
    fire = lambda: launcher(grid, 1, 1, STG(), fn, md, None, None, None,
                            *args_p, *tail)
    for _ in range(5):
        fire()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fire()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    t2 = time.perf_counter()
    return round((t2 - t0) / iters * 1e6, 2)


def native(fn, iters=80):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    t2 = time.perf_counter()
    return round((t2 - t0) / iters * 1e6, 2)


res = {}
for dtype in (torch.float16, torch.bfloat16, torch.float32):
    A = torch.randn((4096, 4096), generator=torch.Generator().manual_seed(2)).to(
        device=DEV, dtype=dtype)
    n = A.numel()
    lim = LIMITS[dtype]
    ptr = A.data_ptr()
    key = str(dtype).split(".")[-1]
    two = dtype is not torch.float32
    bt = 16384 if dtype is torch.float16 else 8192
    d = {"torch_ref": native(lambda: A.nan_to_num_(None, None, None))}
    # current shipped kernel (masked staged for non-fp16, unmasked clamp for fp16)
    if two:
        d["shipped"] = safe(lambda: variant(
            cand._nan_to_num_clamp, (A, n, 0.0, lim, -lim),
            (ptr, n, 0.0, lim, -lim), (bt,), n // bt, {"BLOCK": bt}))
    else:
        d["shipped"] = safe(lambda: variant(
            cand._nan_to_num_stage, (A, n, 0.0, lim, -lim),
            (ptr, n, 0.0, lim, -lim), (bt,), n // bt, {"BLOCK": bt}))
    d["stage_m_I1"] = safe(lambda: variant(
        _stage_loop_m, (A, n, 0.0, lim, -lim), (ptr, n, 0.0, lim, -lim),
        (1, bt), n // bt, {"ITERS": 1, "BLOCK": bt}))
    for it in (2, 4):
        d["stage_m_I%d" % it] = safe(lambda it=it: variant(
            _stage_loop_m, (A, n, 0.0, lim, -lim), (ptr, n, 0.0, lim, -lim),
            (it, bt), n // (it * bt), {"ITERS": it, "BLOCK": bt}))
    if two:
        for it in (2, 4):
            d["clamp_m_I%d" % it] = safe(lambda it=it: variant(
                _clamp_loop_m, (A, n, 0.0, lim, -lim), (ptr, n, 0.0, lim, -lim),
                (it, bt), n // (it * bt), {"ITERS": it, "BLOCK": bt}))
            d["clamp_u_I%d" % it] = safe(lambda it=it: variant(
                _clamp_loop, (A, n, 0.0, lim, -lim), (ptr, n, 0.0, lim, -lim),
                (it, bt), n // (it * bt), {"ITERS": it, "BLOCK": bt}))
        d["clamp_u_I8"] = safe(lambda: variant(
            _clamp_loop, (A, n, 0.0, lim, -lim), (ptr, n, 0.0, lim, -lim),
            (8, bt), n // (8 * bt), {"ITERS": 8, "BLOCK": bt}))
    else:
        d["stage_u_I2"] = safe(lambda: variant(
            _stage_loop_m, (A, n, 0.0, lim, -lim), (ptr, n, 0.0, lim, -lim),
            (2, 4096), n // (2 * 4096), {"ITERS": 2, "BLOCK": 4096}))
    res[key] = d

path = os.path.join(ART, "probe7.json")
with open(path, "w") as fh:
    json.dump(res, fh, indent=1, default=str)
for k, v in res.items():
    print(k, json.dumps(v, default=str))
print("wrote", path)
