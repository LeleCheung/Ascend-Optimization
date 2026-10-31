"""Probe #6: the two untested large-case levers, with correct launcher tails.

  * in-program sequential loop (fewer tasks at the same UB per tile)
  * integer bit-pattern clamp (may need less UB per element -> bigger tile)
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
def _loop(a_ptr, n, nan_v, pos_v, neg_v, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    dt = a_ptr.dtype.element_ty
    base = tl.program_id(0) * (ITERS * BLOCK)
    for i in range(ITERS):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(a_ptr + offs)
        y = tl.maximum(tl.minimum(x, pos_v.to(dt)), neg_v.to(dt))
        y = tl.where(x != x, nan_v.to(dt), y)
        tl.store(a_ptr + offs, y)


@triton.jit
def _bits2(a_ptr, n, nan_v, POSB: tl.constexpr, NEGB: tl.constexpr,
           INFB: tl.constexpr, NEGI: tl.constexpr, BLOCK: tl.constexpr):
    dt = a_ptr.dtype.element_ty
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(a_ptr + offs).to(tl.int16, bitcast=True)
    nanb = nan_v.to(dt).to(tl.int16, bitcast=True)
    mag = v & 32767
    y = tl.where(mag > INFB, nanb, v)
    y = tl.where(v == INFB, POSB, y)
    y = tl.where(v == NEGI, NEGB, y)
    tl.store(a_ptr + offs, y.to(dt, bitcast=True))


@triton.jit
def _bits4(a_ptr, n, nan_v, POSB: tl.constexpr, NEGB: tl.constexpr,
           INFB: tl.constexpr, NEGI: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(a_ptr + offs).to(tl.int32, bitcast=True)
    nanb = nan_v.to(tl.float32).to(tl.int32, bitcast=True)
    mag = v & 2147483647
    y = tl.where(mag > INFB, nanb, v)
    y = tl.where(v == INFB, POSB, y)
    y = tl.where(v == NEGI, NEGB, y)
    tl.store(a_ptr + offs, y.to(tl.float32, bitcast=True))


BITS2 = {torch.float16: dict(POSB=31743, NEGB=-1025, INFB=31744, NEGI=-1024),
         torch.bfloat16: dict(POSB=32639, NEGB=-129, INFB=32640, NEGI=-128)}
BITS4 = dict(POSB=2139095039, NEGB=-8388609, INFB=2139095040, NEGI=-8388608)


def safe(fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "ub overflow" in msg:
            return {"error": "ub_overflow"}
        return {"error": type(exc).__name__ + ": " + msg[:150]}


def variant(jitfn, args_t, args_p, tail, grid, consts, iters=60):
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


def native(fn, iters=60):
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
    d["best_known"] = safe(lambda: variant(
        cand._nan_to_num_clamp if two else cand._nan_to_num_stage,
        (A, n, 0.0, lim, -lim), (ptr, n, 0.0, lim, -lim), (bt,), n // bt,
        {"BLOCK": bt}))
    for iters in (2, 4):
        d["loop%d_B%d" % (iters, bt)] = safe(lambda iters=iters: variant(
            _loop, (A, n, 0.0, lim, -lim), (ptr, n, 0.0, lim, -lim),
            (iters, bt), n // (iters * bt), {"ITERS": iters, "BLOCK": bt,
                                             "num_stages": 1}))
    for blk in (8192, 16384, 32768):
        if two:
            d["bits_B%d" % blk] = safe(lambda blk=blk: variant(
                _bits2, (A, n, 0.0),
                (ptr, n, 0.0), (BITS2[dtype]["POSB"], BITS2[dtype]["NEGB"],
                                BITS2[dtype]["INFB"], BITS2[dtype]["NEGI"], blk),
                n // blk, dict(BLOCK=blk, **BITS2[dtype])))
        else:
            d["bits_B%d" % blk] = safe(lambda blk=blk: variant(
                _bits4, (A, n, 0.0),
                (ptr, n, 0.0), (BITS4["POSB"], BITS4["NEGB"], BITS4["INFB"],
                                BITS4["NEGI"], blk),
                n // blk, dict(BLOCK=blk, **BITS4)))
    # bit exactness including NaN request from the runtime argument
    edge = [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1.5, -3.25,
            65504.0, -65504.0]
    base = torch.tensor(edge * 2, dtype=dtype, device=DEV)
    want = base.clone()
    want.nan_to_num_(None, None, None)
    got = torch.zeros(32, dtype=dtype, device=DEV)
    got[:18] = base
    ref = torch.zeros(32, dtype=dtype, device=DEV)
    ref[:18] = want
    vdt = torch.int16 if two else torch.int32
    try:
        if two:
            _bits2[(1,)](got, 18, 5.0, BLOCK=32, **BITS2[dtype])
        else:
            _bits4[(1,)](got, 18, 5.0, BLOCK=32, **BITS4)
        alt = ref.clone()
        alt.nan_to_num_(5.0, None, None)
        d["bits_exact_default_nan"] = bool(torch.equal(got[:18].view(vdt),
                                                       ref[:18].view(vdt)))
        got2 = torch.zeros(32, dtype=dtype, device=DEV)
        got2[:18] = base
        if two:
            _bits2[(1,)](got2, 18, 5.0, BLOCK=32, **BITS2[dtype])
        else:
            _bits4[(1,)](got2, 18, 5.0, BLOCK=32, **BITS4)
        d["bits_exact_nan5"] = bool(torch.equal(got2[:18].view(vdt),
                                               alt[:18].view(vdt)))
    except Exception as exc:  # noqa: BLE001
        d["bits_exact"] = "%s: %s" % (type(exc).__name__, str(exc)[:150])
    res[key] = d

path = os.path.join(ART, "probe6.json")
with open(path, "w") as fh:
    json.dump(res, fh, indent=1, default=str)
for k, v in res.items():
    print(k, json.dumps(v, default=str))
print("wrote", path)
