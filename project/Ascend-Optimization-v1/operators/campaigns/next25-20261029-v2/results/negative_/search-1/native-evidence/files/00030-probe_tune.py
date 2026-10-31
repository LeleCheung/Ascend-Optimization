"""Tune the device side for the bandwidth-bound sizes with the low-overhead launcher.

Diagnostic only; artifact flushed incrementally.
"""
import json
import os
import time
import traceback

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

import triton
import triton.language as tl
from triton.runtime import driver as tdrv

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
OUT = os.path.join(ART, "tune_probe.json")
DEV = "npu"
res = {}


def flush():
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)


_getstream = torch_npu._C._npu_getCurrentStream
DEV_IDX = torch.npu.current_device()


@triton.jit
def neg_flat(X, n, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(X + offs)
        tl.store(X + offs, -x)
    else:
        m = offs < n
        x = tl.load(X + offs, mask=m)
        tl.store(X + offs, -x, mask=m)


@triton.jit
def neg_gs_even(X, n_iter, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    base = pid * BLOCK
    for _ in range(n_iter):
        offs = base + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        tl.store(X + offs, -x)
        base += step


@triton.jit
def xor_gs_even(X, n_iter, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    base = pid * BLOCK
    for _ in range(n_iter):
        offs = base + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        bits = x.to(tl.int16, bitcast=True) if x.dtype == tl.bfloat16 or x.dtype == tl.float16 else x.to(tl.int32, bitcast=True)
        sign = -32768 if (x.dtype == tl.bfloat16 or x.dtype == tl.float16) else -2147483648
        y = (bits ^ sign)
        yf = y.to(tl.int16, bitcast=True) if (x.dtype == tl.bfloat16 or x.dtype == tl.float16) else y.to(tl.int32, bitcast=True)
        tl.store(X + offs, yf)
        base += step


def bench_wall(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def fast_prep(kern, t, ga, grid, **consts):
    ck = kern.warmup(t, *ga, grid=grid, **consts)
    ck._init_handles()
    return ck


def mk(kern, t, ga, grid, consts_vals):
    """Return a zero-arg callable that fetches the stream and launches."""
    ck = kern.warmup(t, *ga, grid=grid, **(consts_vals if isinstance(consts_vals, dict) else {}))
    ck._init_handles()
    fn = ck.function
    pm = ck.packed_metadata
    launcher = ck._run.launch
    gx = grid[0]
    kargs = tuple(ga)

    def call():
        s = _getstream(DEV_IDX)
        launcher(gx, 1, 1, s, fn, pm, None, None, None, t, *kargs)

    return call


# ---- host floor with the cheap stream getter ---------------------------
try:
    x = torch.randn([1024], device=DEV, dtype=torch.float16)
    ck = neg_flat.warmup(x, 1024, BLOCK=1024, EVEN=True, grid=(1,))
    ck._init_handles()
    fn, pm = ck.function, ck.packed_metadata
    raw = ck._run.launch
    s0 = _getstream(DEV_IDX)
    a = (1, 1, 1, s0, fn, pm, None, None, None, x, 1024, 1024, True)
    res["floor_raw_launch_ms"] = bench_wall(lambda: raw(*a), 500)
    res["floor_stream_getter_ms"] = bench_wall(lambda: _getstream(DEV_IDX), 500)

    def full(xx):
        s = _getstream(DEV_IDX)
        raw(1, 1, 1, s, fn, pm, None, None, None, xx, 1024, 1024, True)

    exp = x.clone()
    torch.neg_(exp)
    full(x)
    torch.npu.synchronize()
    res["floor_full_ok"] = bool(torch.equal(x, exp))
    res["floor_full_n1024_ms"] = bench_wall(lambda: full(x), 500)
    res["torch_n1024_ms"] = bench_wall(lambda: torch.neg_(x), 500)
except Exception:
    res["floor_err"] = traceback.format_exc()[-800:]
flush()

# ---- big-size sweep ---------------------------------------------------
BIG = 16777216
sweep = [("neg", 32, 8192, 2), ("neg", 64, 8192, 2), ("neg", 32, 16384, 2),
         ("neg", 64, 16384, 2), ("neg", 32, 32768, 1), ("neg", 32, 8192, 1),
         ("xor", 32, 8192, 2), ("xor", 64, 16384, 2), ("xor", 32, 32768, 1)]
rows = []
for dt in ("float16", "float32", "bfloat16"):
    t = torch.randn([BIG], device=DEV, dtype=getattr(torch, dt))
    exp = t.clone()
    torch.neg_(exp)
    e = {"dt": dt, "n": BIG, "torch": bench_wall(lambda t=t: torch.neg_(t), 100)}
    for kind, npg, blk, st in sweep:
        kern = neg_gs_even if kind == "neg" else xor_gs_even
        ni = BIG // (npg * blk)
        assert ni * npg * blk == BIG
        try:
            ck = kern.warmup(t, ni, grid=(npg,), BLOCK=blk, num_stages=st)
            ck._init_handles()
            raw = ck._run.launch
            fn, pm = ck.function, ck.packed_metadata
            t.zero_()
            torch.npu.synchronize()
            raw(npg, 1, 1, _getstream(DEV_IDX), fn, pm, None, None, None, t, ni, blk)
            torch.npu.synchronize()
            ok = bool(torch.equal(t, exp))
            e[f"{kind}_{npg}_{blk}_s{st}"] = (bench_wall(
                lambda raw=raw, npg=npg, fn=fn, pm=pm, t=t, ni=ni, blk=blk: raw(
                    npg, 1, 1, _getstream(DEV_IDX), fn, pm, None, None, None, t, ni, blk), 100), ok)
        except Exception:
            e[f"{kind}_{npg}_{blk}_s{st}"] = traceback.format_exc()[-150:]
    rows.append(e)
    res["big"] = rows
    flush()

# ---- mid sizes with the best big config -------------------------------
mid = []
for dt in ("float16", "bfloat16", "float32"):
    for nn, npg, blk in ((1048576, 32, 8192), (1048576, 64, 16384), (65536, 8, 8192),
                         (65536, 4, 16384), (1024, 1, 1024)):
        t = torch.randn([nn], device=DEV, dtype=getattr(torch, dt))
        exp = t.clone()
        torch.neg_(exp)
        try:
            ni = nn // (npg * blk)
            assert ni * npg * blk == nn
            ck = neg_gs_even.warmup(t, ni, grid=(npg,), BLOCK=blk, num_stages=2)
            ck._init_handles()
            raw, fn, pm = ck._run.launch, ck.function, ck.packed_metadata
            t.zero_()
            torch.npu.synchronize()
            raw(npg, 1, 1, _getstream(DEV_IDX), fn, pm, None, None, None, t, ni, blk)
            torch.npu.synchronize()
            ok = bool(torch.equal(t, exp))
            w = bench_wall(lambda raw=raw, npg=npg, fn=fn, pm=pm, t=t, ni=ni, blk=blk: raw(
                npg, 1, 1, _getstream(DEV_IDX), fn, pm, None, None, None, t, ni, blk), 200)
            mid.append({"dt": dt, "n": nn, "npg": npg, "blk": blk, "wall": w, "ok": ok,
                        "torch": bench_wall(lambda t=t: torch.neg_(t), 300)})
        except Exception:
            mid.append({"dt": dt, "n": nn, "npg": npg, "blk": blk, "err": traceback.format_exc()[-150:]})
res["mid"] = mid
flush()
print(json.dumps(res, indent=1)[:9000])
