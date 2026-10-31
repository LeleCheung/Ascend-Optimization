"""Capture the exact launcher args used by the working JIT path; time direct replay.

Diagnostic only: monkeypatch is process-local and used solely to observe the
arguments that the normal (working) launch passes to the C launcher.
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
from triton.backends.ascend.driver import NPULauncher

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
DEV = "npu"
res = {}


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
def neg_gs(X, n, n_iter, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    step = npg * BLOCK
    base = pid * BLOCK
    for _ in range(n_iter):
        offs = base + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(X + offs, mask=m)
        tl.store(X + offs, -x, mask=m)
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


def bench_dev_percall(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    evs = [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in evs:
        s.record()
        fn()
        e.record()
    torch.npu.synchronize()
    return sum(s.elapsed_time(e) for s, e in evs) / iters


x = torch.randn([1024], device=DEV, dtype=torch.float16)
n = 1024

# ---- capture the args of one normal JIT launch -----------------------------
cap = {}
orig_call = NPULauncher.__call__


def spy(self, *args, **kwargs):
    if "args" not in cap:
        cap["nargs"] = len(args)
        cap["types"] = [type(a).__name__ for a in args]
        cap["ints"] = [a if isinstance(a, (int, float, type(None))) else None for a in args]
        cap["reprs"] = [repr(a)[:100] for a in args]
        cap["kwargs"] = {k: repr(v)[:60] for k, v in kwargs.items()}
        cap["args"] = args
    return orig_call(self, *args, **kwargs)


NPULauncher.__call__ = spy
try:
    neg_flat[(1,)](x, n, BLOCK=1024, EVEN=True)
    torch.npu.synchronize()
    res["capture"] = {k: v for k, v in cap.items() if k != "args"}
except Exception:
    res["capture"] = traceback.format_exc()[-1500:]
finally:
    NPULauncher.__call__ = orig_call

truth = x.clone()
torch.neg_(truth)

# ---- full JIT timing ------------------------------------------------------
tim = {}
tim["jit_1024"] = bench_wall(lambda: neg_flat[(1,)](x, n, BLOCK=1024, EVEN=True), 500)
tim["jit_1024_dev"] = bench_dev_percall(lambda: neg_flat[(1,)](x, n, BLOCK=1024, EVEN=True), 200)
tim["torch_neg_"] = bench_wall(lambda: torch.neg_(x), 500)

# ---- direct replay of the captured launcher args --------------------------
try:
    from triton.runtime import driver as _drv
    ck = neg_flat.warmup(x, n, BLOCK=1024, EVEN=True, grid=(1,))
    ck._init_handles()
    launcher = ck.run
    cargs = cap["args"]
    res["cargs_len"] = len(cargs)
    # sanity: replaying captured args must launch successfully and negate once
    x.zero_()
    x.fill_(3.0)
    torch.npu.synchronize()
    launcher(*cargs)
    torch.npu.synchronize()
    res["replay_ok"] = bool(torch.equal(x, torch.full_like(x, -3.0)))
    tim["replay_1024"] = bench_wall(lambda: launcher(*cargs), 500)
    tim["replay_1024_dev"] = bench_dev_percall(lambda: launcher(*cargs), 200)

    # rebuild the arg vector explicitly (as a candidate would)
    stream = _drv.active.get_current_stream(_drv.active.get_current_device())
    res["explicit_stream"] = stream
    keep = []
    for i, a in enumerate(cargs):
        if isinstance(a, int) and i >= 9:
            keep.append(a)
    res["trailing_ints"] = keep
    explicit = list(cargs[:9]) + keep + [x, n]
    res["explicit_len"] = len(explicit)
    tim["stream_fetch"] = bench_wall(
        lambda: _drv.active.get_current_stream(_drv.active.get_current_device()), 500)
    tim["npu_current_stream"] = bench_wall(lambda: torch.npu.current_stream().npu_stream, 500)
    # explicit rebuild: replace stream + tensor in the captured vector
    try:
        ex = list(cargs)
        ex[3] = stream
        launcher(*ex)
        torch.npu.synchronize()
        res["explicit_ok"] = True
        tim["explicit_1024"] = bench_wall(lambda: launcher(*ex), 500)
    except Exception:
        res["explicit_ok"] = traceback.format_exc()[-400:]
except Exception:
    res["direct"] = traceback.format_exc()[-1500:]

# ---- device-only cost of real sizes --------------------------------------
fitems = [("float16", [1024]), ("float16", [65536]), ("float16", [1048576]),
          ("float16", [4096, 4096]), ("float32", [1024]), ("float32", [4096, 4096]),
          ("bfloat16", [1048576])]
dev = []
for dt, shape in fitems:
    t = torch.randn(shape, device=DEV, dtype=getattr(torch, dt))
    nn = t.numel()
    e = {"dt": dt, "n": nn}
    e["torch_dev"] = bench_dev_percall(lambda t=t: torch.neg_(t), 100)
    for npg, blk in ((20, 8192), (40, 8192), (40, 4096), (80, 2048)):
        try:
            ni = (nn + blk * npg - 1) // (blk * npg)
            e[f"gs_{npg}_{blk}_dev"] = bench_dev_percall(
                lambda t=t, p=npg, b=blk, k=ni: neg_gs[(p,)](t, t.numel(), k, BLOCK=b), 100)
        except Exception:
            e[f"gs_{npg}_{blk}_dev"] = traceback.format_exc()[-200:]
    for blk in (2048, 4096, 8192):
        try:
            g = ((nn + blk - 1) // blk,)
            e[f"flat_{blk}_dev"] = bench_dev_percall(
                lambda t=t, b=blk, g=g: neg_flat[g](t, t.numel(), BLOCK=b, EVEN=(t.numel() % b == 0)), 100)
        except Exception:
            e[f"flat_{blk}_dev"] = traceback.format_exc()[-200:]
    dev.append(e)
res["device_only"] = dev

# ---- masked bf16 1025 compiles at which BLOCK? ---------------------------
ub = {}
for blk in (512, 1024, 2048, 4096):
    t = torch.randn([1025], device=DEV, dtype=torch.bfloat16)
    try:
        neg_flat[((1025 + blk - 1) // blk,)](t, 1025, BLOCK=blk, EVEN=False)
        torch.npu.synchronize()
        ub[blk] = "ok"
    except Exception:
        ub[blk] = traceback.format_exc()[-160:]
res["bf16_1025_masked"] = ub

res["timing_ms"] = tim
with open(os.path.join(ART, "replay_probe.json"), "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps({"timing_ms": tim, "capture": res.get("capture"), "replay_ok": res.get("replay_ok"),
                  "cargs_len": res.get("cargs_len"), "trailing_ints": res.get("trailing_ints"),
                  "bf16_1025_masked": ub, "explicit_ok": res.get("explicit_ok")}, indent=1))
