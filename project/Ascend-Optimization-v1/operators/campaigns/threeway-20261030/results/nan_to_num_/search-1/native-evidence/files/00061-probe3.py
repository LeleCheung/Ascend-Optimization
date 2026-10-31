"""Probe #3: device-bound throughput of large-case kernel variants (prebuilt
launch + trailing sync), native calibration, and slim-launch host cost.

Kernels all share the signature (a_ptr, n, nan_v, pos_v, neg_v, BLOCK) so one
argument tuple drives every prebuilt launch.
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
def _pair(a_ptr, n, nan_v, pos_v, neg_v, SUB: tl.constexpr):
    dt = a_ptr.dtype.element_ty
    off0 = tl.program_id(0) * (2 * SUB) + tl.arange(0, SUB)
    off1 = off0 + SUB
    x0 = tl.load(a_ptr + off0)
    x1 = tl.load(a_ptr + off1)
    y0 = tl.maximum(tl.minimum(x0, pos_v.to(dt)), neg_v.to(dt))
    y1 = tl.maximum(tl.minimum(x1, pos_v.to(dt)), neg_v.to(dt))
    y0 = tl.where(x0 != x0, nan_v.to(dt), y0)
    y1 = tl.where(x1 != x1, nan_v.to(dt), y1)
    tl.store(a_ptr + off0, y0)
    tl.store(a_ptr + off1, y1)


@triton.jit
def _quad(a_ptr, n, nan_v, pos_v, neg_v, SUB: tl.constexpr):
    dt = a_ptr.dtype.element_ty
    off0 = tl.program_id(0) * (4 * SUB) + tl.arange(0, SUB)
    off1 = off0 + SUB
    off2 = off1 + SUB
    off3 = off2 + SUB
    x0 = tl.load(a_ptr + off0)
    x1 = tl.load(a_ptr + off1)
    x2 = tl.load(a_ptr + off2)
    x3 = tl.load(a_ptr + off3)
    y0 = tl.maximum(tl.minimum(x0, pos_v.to(dt)), neg_v.to(dt))
    y1 = tl.maximum(tl.minimum(x1, pos_v.to(dt)), neg_v.to(dt))
    y2 = tl.maximum(tl.minimum(x2, pos_v.to(dt)), neg_v.to(dt))
    y3 = tl.maximum(tl.minimum(x3, pos_v.to(dt)), neg_v.to(dt))
    y0 = tl.where(x0 != x0, nan_v.to(dt), y0)
    y1 = tl.where(x1 != x1, nan_v.to(dt), y1)
    y2 = tl.where(x2 != x2, nan_v.to(dt), y2)
    y3 = tl.where(x3 != x3, nan_v.to(dt), y3)
    tl.store(a_ptr + off0, y0)
    tl.store(a_ptr + off1, y1)
    tl.store(a_ptr + off2, y2)
    tl.store(a_ptr + off3, y3)


@triton.jit
def _slim(a_ptr, nan_v, POS: tl.constexpr, NEG: tl.constexpr, BLOCK: tl.constexpr):
    dt = a_ptr.dtype.element_ty
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a_ptr + offs)
    y = tl.maximum(tl.minimum(x, POS), NEG)
    tl.store(a_ptr + offs, tl.where(x != x, nan_v.to(dt), y))


def safe(fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        short = "ub_overflow" if "ub overflow" in msg else msg[:200]
        return {"error": type(exc).__name__ + ": " + short}


def device_us(jitfn, A, args_t, args_p, grid, consts):
    """Prebuilt direct launch; per-iteration time including the drain sync."""
    ck = jitfn.warmup(A, *args_t, grid=(1,), **consts)
    launcher, fn, md = ck.run.launch, ck.function, ck.packed_metadata
    block = consts.get("BLOCK", consts.get("SUB"))
    args = args_p
    fire = lambda: launcher(grid, 1, 1, STG(), fn, md, None, None, None,
                            *args, block)
    for _ in range(5):
        fire()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(60):
        fire()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    t2 = time.perf_counter()
    return {"host_us": (t1 - t0) / 60 * 1e6, "dev_us": (t2 - t0) / 60 * 1e6}


def host_us(fn, iters=300):
    for _ in range(30):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e6


res = {}
for dtype in (torch.float16, torch.bfloat16, torch.float32):
    A = torch.randn((4096, 4096), generator=torch.Generator().manual_seed(2)).to(
        device=DEV, dtype=dtype)
    B = torch.empty_like(A)
    n = A.numel()
    lim = LIMITS[dtype]
    args5 = (A.data_ptr(), n, 0.0, lim, -lim)
    args5_t = (A, n, 0.0, lim, -lim)
    key = str(dtype).split(".")[-1]
    d = {}
    # native calibration with the same measurement shape
    def native(fn):
        for _ in range(5):
            fn()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(60):
            fn()
        t1 = time.perf_counter()
        torch.npu.synchronize()
        t2 = time.perf_counter()
        return {"host_us": (t1 - t0) / 60 * 1e6, "dev_us": (t2 - t0) / 60 * 1e6}
    d["copy_"] = safe(lambda: native(lambda: A.copy_(B)))
    d["torch_ref"] = safe(lambda: native(lambda: A.nan_to_num_(None, None, None)))
    d["clamp8192"] = safe(lambda: device_us(cand._nan_to_num_clamp, A, args5_t,
                                            args5, n // 8192, {"BLOCK": 8192}))
    d["clamp16384"] = safe(lambda: device_us(cand._nan_to_num_clamp, A, args5_t,
                                             args5, n // 16384, {"BLOCK": 16384}))
    d["stage8192"] = safe(lambda: device_us(cand._nan_to_num_stage, A, args5_t,
                                            args5, n // 8192, {"BLOCK": 8192}))
    sub = 4096 if dtype is torch.float32 else 8192
    d["pair_SUB%d" % sub] = safe(lambda: device_us(_pair, A, args5_t, args5,
                                                  n // (2 * sub), {"SUB": sub}))
    sub2 = 2048 if dtype is torch.float32 else 4096
    d["quad_SUB%d" % sub2] = safe(lambda: device_us(_quad, A, args5_t, args5,
                                                    n // (4 * sub2), {"SUB": sub2}))
    sub3 = 8192 if dtype is torch.float32 else 16384
    d["pair_SUB%d" % sub3] = safe(lambda: device_us(_pair, A, args5_t, args5,
                                                    n // (2 * sub3), {"SUB": sub3}))
    # seed run for reference in this process
    cand.run(A)
    torch.npu.synchronize()
    d["seed_run"] = native(lambda: cand.run(A))
    res[key] = d

# ---- slim launch host cost on a tiny tensor (2 runtime args vs 5) ----------
A = torch.randn(1024, generator=torch.Generator().manual_seed(0)).to(
    device=DEV, dtype=torch.float16)
n = A.numel()
lim = LIMITS[torch.float16]
slim = {}
try:
    ck = _slim.warmup(A, 0.0, POS=lim, NEG=-lim, BLOCK=1024, grid=(1,))
    sl, sfn, smd = ck.run.launch, ck.function, ck.packed_metadata
    ptr = A.data_ptr()
    slim["slim_2runtime"] = host_us(lambda: sl(1, 1, 1, STG(), sfn, smd,
                                               None, None, None, ptr, 0.0, lim, lim, 1024))
    ck2 = cand._nan_to_num_clamp.warmup(A, n, 0.0, lim, -lim, BLOCK=1024, grid=(1,))
    l2, f2, m2 = ck2.run.launch, ck2.function, ck2.packed_metadata
    slim["clamp_5runtime"] = host_us(lambda: l2(1, 1, 1, STG(), f2, m2,
                                                None, None, None, ptr, n, 0.0,
                                                lim, -lim, 1024))
    slim["slim_2runtime_again"] = host_us(lambda: sl(1, 1, 1, STG(), sfn, smd,
                                                     None, None, None, ptr, 0.0,
                                                     lim, lim, 1024))
    slim["torch_ref"] = host_us(lambda: A.nan_to_num_(None, None, None))
except Exception as exc:  # noqa: BLE001
    slim["error"] = "%s: %s" % (type(exc).__name__, exc)
res["slim_host_us"] = slim

path = os.path.join(ART, "probe3.json")
with open(path, "w") as fh:
    json.dump(res, fh, indent=1, default=str)
for k, v in res.items():
    print(k, json.dumps(v, default=str))
print("wrote", path)
