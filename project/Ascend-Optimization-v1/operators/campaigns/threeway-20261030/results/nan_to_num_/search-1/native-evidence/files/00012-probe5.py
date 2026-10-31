"""Probe #5: device-side sweep for the three 4096x4096 cases.

Levers that do not change semantics: num_warps / num_stages, in-program loop over
sequential tiles (fewer tasks, same UB), and a bit-pattern (int16/int32) clamp
that may need less UB per element than the float path. All measured with a
prebuilt direct launch plus drain sync.
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


BITS = {torch.float16: dict(POSB=31743, NEGB=-1025, INFB=31744, NEGI=-1024),
        torch.bfloat16: dict(POSB=32639, NEGB=-129, INFB=32640, NEGI=-128)}


def safe(fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "ub overflow" in msg:
            return {"error": "ub_overflow"}
        return {"error": type(exc).__name__ + ": " + msg[:150]}


def run_variant(jitfn, A, args_t, args_p, grid, consts, iters=60):
    ck = jitfn.warmup(*args_t, grid=(1,), **consts)
    launcher, fn, md = ck.run.launch, ck.function, ck.packed_metadata
    tail = consts.get("BLOCK", consts.get("SUB", ck.metadata.num_warps))
    ub = getattr(ck.metadata, "required_ub_bits", "?")
    fire = lambda: launcher(grid, 1, 1, STG(), fn, md, None, None, None,
                            *args_p, tail)
    for _ in range(5):
        fire()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fire()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    t2 = time.perf_counter()
    return {"dev_us": round((t2 - t0) / iters * 1e6, 2), "ub_bits": ub}


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
    args_p = (A.data_ptr(), n, 0.0, lim, -lim)
    args_t = (A, n, 0.0, lim, -lim)
    key = str(dtype).split(".")[-1]
    two = dtype is not torch.float32
    d = {"torch_ref": native(lambda: A.nan_to_num_(None, None, None))}

    base_kern = cand._nan_to_num_clamp if two else cand._nan_to_num_stage
    base_block = 16384 if dtype is torch.float16 else 8192
    d["baseline_B%d" % base_block] = safe(lambda: run_variant(
        base_kern, A, args_t, args_p, n // base_block, {"BLOCK": base_block}))
    for nw in (8, 16, 64):
        d["baseline_B%d_nw%d" % (base_block, nw)] = safe(lambda nw=nw: run_variant(
            base_kern, A, args_t, args_p, n // base_block,
            {"BLOCK": base_block, "num_warps": nw}))
    # in-program sequential loop: fewer tasks, same UB per tile
    lb = 16384 if dtype is torch.float16 else 8192
    for iters, ns in ((2, 1), (4, 1), (2, 2)):
        d["loop%d_B%d_ns%d" % (iters, lb, ns)] = safe(lambda iters=iters, ns=ns,
                                                      lb=lb: run_variant(
            _loop, A, args_t, args_p, n // (iters * lb),
            {"ITERS": iters, "BLOCK": lb, "num_stages": ns}))
    # bigger tiles under friendlier compile options
    if dtype is torch.float16:
        d["clamp32768_ns1"] = safe(lambda: run_variant(
            cand._nan_to_num_clamp, A, args_t, args_p, n // 32768,
            {"BLOCK": 32768, "num_stages": 1}))
        d["clamp32768_nw8"] = safe(lambda: run_variant(
            cand._nan_to_num_clamp, A, args_t, args_p, n // 32768,
            {"BLOCK": 32768, "num_warps": 8}))
    else:
        d["clamp16384_ns1"] = safe(lambda: run_variant(
            cand._nan_to_num_clamp, A, args_t, args_p, n // 16384,
            {"BLOCK": 16384, "num_stages": 1}))
    if two:
        b = BITS[dtype]
        for blk in (8192, 16384):
            d["bits_B%d" % blk] = safe(lambda blk=blk: run_variant(
                _bits2, A, args_t, args_p, n // blk,
                dict(BLOCK=blk, **BITS[dtype])))
        # bit-exactness on special values
        edge = [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1.5, -3.25,
                65504.0, -65504.0]
        base = torch.tensor(edge * 2, dtype=dtype, device=DEV)
        want = base.clone()
        want.nan_to_num_(None, None, None)
        got = torch.zeros(32, dtype=dtype, device=DEV)
        got[:18] = base
        ref = torch.zeros(32, dtype=dtype, device=DEV)
        ref[:18] = want
        try:
            _bits2[(1,)](got, 18, 0.0, BLOCK=32, **BITS[dtype])
            d["bits_bit_exact"] = bool(torch.equal(got.view(torch.int16),
                                                   ref.view(torch.int16)))
        except Exception as exc:  # noqa: BLE001
            d["bits_bit_exact"] = safe(lambda: (_ for _ in ()).throw(exc))
    res[key] = d

path = os.path.join(ART, "probe5.json")
with open(path, "w") as fh:
    json.dump(res, fh, indent=1, default=str)
for k, v in res.items():
    print(k, json.dumps(v, default=str))
print("wrote", path)
