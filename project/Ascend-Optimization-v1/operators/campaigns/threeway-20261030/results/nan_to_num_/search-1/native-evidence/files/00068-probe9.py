"""Probe #9: push the device-bound regime further.

  * more sequential tiles per program (ITERS 8/16) at the same tile
  * two-stage software pipelining: num_stages=2 with the tile halved so the
    doubled buffer still fits the 192 KiB UB budget
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


def safe(fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "ub overflow" in msg:
            return {"error": "ub_overflow"}
        return {"error": type(exc).__name__ + ": " + msg[:120]}


def variant(jitfn, args_t, args_p, tail, grid, consts, iters=100):
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


def native(fn, iters=100):
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
    args_t = (A, n, 0.0, lim, -lim)
    args_p = (ptr, n, 0.0, lim, -lim)
    key = str(dtype).split(".")[-1]
    two = dtype is not torch.float32
    kname = _clamp_loop if two else _stage_loop
    kern = cand._ntn_clamp_loop if two else cand._ntn_stage_loop
    base_tile = 16384 if two else 8192
    d = {"torch_ref": native(lambda: A.nan_to_num_(None, None, None))}

    def go(tile, it, ns):
        grid = -(-n // (it * tile))
        return variant(kern, args_t, args_p, (it, tile), grid,
                       {"ITERS": it, "BLOCK": tile, "num_stages": ns})

    d["tile%d_I1_ns1" % base_tile] = safe(lambda: go(base_tile, 1, 1))
    for it in (2, 4, 8, 16):
        d["tile%d_I%d_ns1" % (base_tile, it)] = safe(
            lambda it=it: go(base_tile, it, 1))
    half = base_tile // 2
    for it in (2, 4, 8):
        d["tile%d_I%d_ns2" % (half, it)] = safe(lambda it=it: go(half, it, 2))
    third = base_tile // 4
    for it in (2, 8):
        d["tile%d_I%d_ns2" % (third, it)] = safe(lambda it=it: go(third, it, 2))
        d["tile%d_I%d_ns4" % (third, it)] = safe(lambda it=it: go(third, it, 4))
    res[key] = d

path = os.path.join(ART, "probe9.json")
with open(path, "w") as fh:
    json.dump(res, fh, indent=1, default=str)
for k, v in res.items():
    print(k, json.dumps(v, default=str))
print("wrote", path)
