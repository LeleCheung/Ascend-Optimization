"""Corrected tuning probe: validate cheap stream getter, sweep big sizes, test xor-negate."""
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
OUT = os.path.join(ART, "tune2_probe.json")
DEV = "npu"
res = {}


def flush():
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)


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
def xor16_gs_even(X, n_iter, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    base = pid * BLOCK
    for _ in range(n_iter):
        offs = base + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        y = x.to(tl.int16, bitcast=True) ^ -32768
        tl.store(X + offs, y.to(tl.int16, bitcast=True))
        base += step


@triton.jit
def xor32_gs_even(X, n_iter, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    base = pid * BLOCK
    for _ in range(n_iter):
        offs = base + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        y = x.to(tl.int32, bitcast=True) ^ -2147483648
        tl.store(X + offs, y.to(tl.int32, bitcast=True))
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


# ---- stage A: stream getter equivalence -------------------------------
try:
    good = tdrv.active.get_current_stream(None)
    dev_idx = torch.npu.current_device()
    raw_s = torch_npu._C._npu_getCurrentStream(dev_idx)
    info = {"driver": int(good), "dev_idx_type": type(dev_idx).__name__,
            "dev_idx": getattr(dev_idx, "__len__", lambda: -1)() if hasattr(dev_idx, "__len__") else dev_idx,
            "cheap_type": type(raw_s).__name__, "cheap_repr": repr(raw_s)[:120]}
    if isinstance(raw_s, (tuple, list)):
        info["cheap_first"] = int(raw_s[0])
        info["matches"] = int(raw_s[0]) == int(good)
        S_GET = lambda: int(raw_s[0]) if False else None
    else:
        info["matches"] = int(raw_s) == int(good)
    if info.get("matches"):
        if isinstance(raw_s, (tuple, list)):
            def getstream():
                return int(torch_npu._C._npu_getCurrentStream(dev_idx)[0])
        else:
            def getstream():
                return int(torch_npu._C._npu_getCurrentStream(dev_idx))
        info["getstream_ms"] = bench_wall(getstream, 1000)
    else:
        def getstream():
            return int(tdrv.active.get_current_stream(None))
        info["getstream_ms_driver"] = bench_wall(getstream, 500)
    res["stream"] = info
except Exception:
    res["stream"] = traceback.format_exc()[-800:]
    def getstream():
        return int(tdrv.active.get_current_stream(None))
flush()

# ---- floor with full per-call arg build ------------------------------
try:
    x = torch.randn([1024], device=DEV, dtype=torch.float16)
    ck = neg_flat.warmup(x, 1024, BLOCK=1024, EVEN=True, grid=(1,))
    ck._init_handles()
    fn, pm, raw = ck.function, ck.packed_metadata, ck._run.launch
    exp = x.clone()
    torch.neg_(exp)
    x.zero_()
    torch.npu.synchronize()
    raw(1, 1, 1, getstream(), fn, pm, None, None, None, x, 1024, 1024, True)
    torch.npu.synchronize()
    res["floor_ok"] = bool(torch.equal(x, exp))

    def full(xx):
        s = getstream()
        raw(1, 1, 1, s, fn, pm, None, None, None, xx, 1024, 1024, True)

    res["floor_n1024_ms"] = bench_wall(lambda: full(x), 500)
    res["torch_n1024_ms"] = bench_wall(lambda: torch.neg_(x), 500)
except Exception:
    res["floor_err"] = traceback.format_exc()[-800:]
flush()

# ---- stage B: big-size sweep ----------------------------------------
BIG = 16777216


def run_cfg(kern, t, nn, npg, blk, st):
    ni = nn // (npg * blk)
    if ni * npg * blk != nn:
        raise ValueError("not divisible")
    ck = kern.warmup(t, ni, grid=(npg,), BLOCK=blk, num_stages=st)
    ck._init_handles()
    fn, pm, raw = ck.function, ck.packed_metadata, ck._run.launch
    t.zero_()
    torch.npu.synchronize()
    raw(npg, 1, 1, getstream(), fn, pm, None, None, None, t, ni, blk)
    torch.npu.synchronize()
    return ck, ni


PLAN = {
    "float16": [("neg", 32, 8192, 2), ("neg", 64, 8192, 2), ("neg", 32, 16384, 2),
                ("neg", 64, 16384, 2), ("neg", 32, 32768, 1), ("xor16", 32, 8192, 2),
                ("xor16", 32, 16384, 2)],
    "bfloat16": [("neg", 32, 8192, 2), ("neg", 32, 16384, 2), ("neg", 64, 8192, 2),
                 ("xor16", 32, 8192, 2), ("xor16", 32, 16384, 2), ("xor16", 64, 8192, 2)],
    "float32": [("neg", 32, 8192, 2), ("neg", 32, 16384, 2), ("neg", 64, 8192, 2),
                ("xor32", 32, 8192, 2), ("xor32", 32, 16384, 2), ("neg", 32, 8192, 1)],
}
KERN = {"neg": neg_gs_even, "xor16": xor16_gs_even, "xor32": xor32_gs_even}
rows = []
for dt, cfgs in PLAN.items():
    t = torch.randn([BIG], device=DEV, dtype=getattr(torch, dt))
    exp = t.clone()
    torch.neg_(exp)
    e = {"dt": dt, "n": BIG, "torch": bench_wall(lambda t=t: torch.neg_(t), 100)}
    for kind, npg, blk, st in cfgs:
        key = f"{kind}_{npg}_{blk}_s{st}"
        try:
            ck, ni = run_cfg(KERN[kind], t, BIG, npg, blk, st)
            ok = bool(torch.equal(t, exp))
            fn, pm, raw = ck.function, ck.packed_metadata, ck._run.launch
            e[key] = [bench_wall(lambda: raw(npg, 1, 1, getstream(), fn, pm, None, None, None, t, ni, blk), 100), ok]
        except Exception:
            e[key] = traceback.format_exc()[-160:]
    rows.append(e)
    res["big"] = rows
    flush()

# ---- stage C: mid sizes ---------------------------------------------
mid = []
for dt in ("float16", "bfloat16", "float32"):
    xor_kind = "xor16" if dt != "float32" else "xor32"
    for nn, npg, blk, kind in ((1048576, 32, 8192, "neg"), (1048576, 64, 16384, "neg"),
                               (65536, 8, 8192, "neg"), (1024, 1, 1024, "neg"),
                               (1048576, 32, 8192, xor_kind), (65536, 8, 8192, xor_kind)):
        t = torch.randn([nn], device=DEV, dtype=getattr(torch, dt))
        exp = t.clone()
        torch.neg_(exp)
        rec = {"dt": dt, "n": nn, "npg": npg, "blk": blk, "kind": kind}
        try:
            ck, ni = run_cfg(KERN[kind], t, nn, npg, blk, 2)
            rec["ok"] = bool(torch.equal(t, exp))
            fn, pm, raw = ck.function, ck.packed_metadata, ck._run.launch
            rec["wall"] = bench_wall(lambda: raw(npg, 1, 1, getstream(), fn, pm, None, None, None, t, ni, blk), 300)
            rec["torch"] = bench_wall(lambda t=t: torch.neg_(t), 300)
        except Exception:
            rec["err"] = traceback.format_exc()[-160:]
        mid.append(rec)
res["mid"] = mid
flush()
print(json.dumps(res, indent=1)[:9000])
