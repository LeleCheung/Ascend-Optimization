"""Probe #4: large-case device throughput for clamp/stage/pair/quad variants and
num_stages=1 bigger tiles, measured with prebuilt launches + drain sync.
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


def safe(fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        short = "ub_overflow" if "ub overflow" in msg else msg[:160]
        return {"error": type(exc).__name__ + ": " + short}


def variant(jitfn, A, args_t, args_p, grid, consts, iters=60):
    ck = jitfn.warmup(*args_t, grid=(1,), **consts)
    launcher, fn, md = ck.run.launch, ck.function, ck.packed_metadata
    block = consts.get("BLOCK", consts.get("SUB"))
    fire = lambda: launcher(grid, 1, 1, STG(), fn, md, None, None, None,
                            *args_p, block)
    for _ in range(5):
        fire()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fire()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    t2 = time.perf_counter()
    return {"host_us": round((t1 - t0) / iters * 1e6, 2),
            "dev_us": round((t2 - t0) / iters * 1e6, 2)}


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
    return {"host_us": round((t1 - t0) / iters * 1e6, 2),
            "dev_us": round((t2 - t0) / iters * 1e6, 2)}


res = {}
for dtype in (torch.float16, torch.bfloat16, torch.float32):
    A = torch.randn((4096, 4096), generator=torch.Generator().manual_seed(2)).to(
        device=DEV, dtype=dtype)
    B = torch.empty(A.shape, dtype=dtype, device=DEV)
    B.copy_(A)
    n = A.numel()
    lim = LIMITS[dtype]
    args_p = (A.data_ptr(), n, 0.0, lim, -lim)
    args_t = (A, n, 0.0, lim, -lim)
    key = str(dtype).split(".")[-1]
    two = dtype is not torch.float32
    d = {}
    d["copy_"] = safe(lambda: native(lambda: A.copy_(B)))
    d["torch_ref"] = safe(lambda: native(lambda: A.nan_to_num_(None, None, None)))
    d["clamp8192"] = safe(lambda: variant(cand._nan_to_num_clamp, A, args_t, args_p,
                                          n // 8192, {"BLOCK": 8192}))
    d["clamp16384"] = safe(lambda: variant(cand._nan_to_num_clamp, A, args_t, args_p,
                                           n // 16384, {"BLOCK": 16384}))
    d["stage8192"] = safe(lambda: variant(cand._nan_to_num_stage, A, args_t, args_p,
                                          n // 8192, {"BLOCK": 8192}))
    sub = 8192 if two else 4096
    d["pair%d" % sub] = safe(lambda: variant(_pair, A, args_t, args_p, n // (2 * sub),
                                             {"SUB": sub}))
    qsub = 4096 if two else 2048
    d["quad%d" % qsub] = safe(lambda: variant(_quad, A, args_t, args_p,
                                              n // (4 * qsub), {"SUB": qsub}))
    big = 32768 if two else 16384
    d["clamp%d_ns1" % big] = safe(lambda: variant(cand._nan_to_num_clamp, A, args_t,
                                                  args_p, n // big,
                                                  {"BLOCK": big, "num_stages": 1}))
    psub = 16384 if two else 8192
    d["pair%d_ns1" % psub] = safe(lambda: variant(_pair, A, args_t, args_p,
                                                  n // (2 * psub),
                                                  {"SUB": psub, "num_stages": 1}))
    cand.run(A)
    torch.npu.synchronize()
    d["seed_run"] = native(lambda: cand.run(A))
    res[key] = d

# accuracy sanity of the fastest structural candidates on special values
chk = {}
try:
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        lim = LIMITS[dtype]
        vdt = torch.int32 if dtype is torch.float32 else torch.int16
        edge = [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1.5, -3.25,
                65504.0, -65504.0]
        base = torch.tensor(edge * 2, dtype=dtype, device=DEV)  # 18 elements
        want = base.clone()
        want.nan_to_num_(None, None, None)
        for name, kern, consts, cov in (("pair", _pair, {"SUB": 16}, 32),
                                        ("quad", _quad, {"SUB": 8}, 32),
                                        ("clamp", cand._nan_to_num_clamp,
                                         {"BLOCK": 32}, 32)):
            t = torch.zeros(cov, dtype=dtype, device=DEV)
            t[:18] = base
            ref = torch.zeros(cov, dtype=dtype, device=DEV)
            ref[:18] = want
            kern[(1,)](t, t.numel(), 0.0, lim, -lim, **consts)
            chk["%s_%s" % (str(dtype).split(".")[-1], name)] = bool(
                torch.equal(t.view(vdt), ref.view(vdt)))
except Exception as exc:  # noqa: BLE001
    chk["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:300])
res["special_bitcheck"] = chk

path = os.path.join(ART, "probe4.json")
with open(path, "w") as fh:
    json.dump(res, fh, indent=1, default=str)
for k, v in res.items():
    print(k, json.dumps(v, default=str))
print("wrote", path)
