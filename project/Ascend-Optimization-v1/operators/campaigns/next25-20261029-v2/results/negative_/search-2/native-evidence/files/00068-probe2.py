"""Bounded diagnosis #2: decompose the ~50us Triton launch floor and sweep
device-side tiling for the 4096x4096 cases.  Diagnostics only.
"""

import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main as cand  # noqa: E402

ART = Path(os.environ.get("KGS_DEBUG_ARTIFACTS", ".")) / "probe2.json"
DEV = "npu:0"
out = {"launch": {}, "sweep": {}, "notes": []}


def bench(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


@triton.jit
def _k0():
    pid = tl.program_id(0)


@triton.jit
def _k1(X):
    pid = tl.program_id(0)


@triton.jit
def _k2(X, n):
    pid = tl.program_id(0)


@triton.jit
def _k3(X, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)


@triton.jit
def _k4(X, n, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)


@triton.jit
def _neg_b(X, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    v = tl.load(X + o, mask=m)
    tl.store(X + o, -v, mask=m)


@triton.jit
def _neg_nconst(X, BLOCK: tl.constexpr, N: tl.constexpr):
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    v = tl.load(X + o, mask=m)
    tl.store(X + o, -v, mask=m)


@triton.jit
def _neg_gs(X, n, BLOCK: tl.constexpr, ITERS: tl.constexpr, STEP: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK
    for i in tl.static_range(ITERS):
        o = base + i * STEP + tl.arange(0, BLOCK)
        m = o < n
        v = tl.load(X + o, mask=m)
        tl.store(X + o, -v, mask=m)


def main():
    a = torch.zeros(1024, dtype=torch.float16, device=DEV)
    big32 = torch.zeros((4096, 4096), dtype=torch.float32, device=DEV)
    big16 = torch.zeros((4096, 4096), dtype=torch.float16, device=DEV)

    # ---------- launch-path decomposition on a 1024-element tensor ----------
    L = out["launch"]
    L["torch_neg_"] = bench(lambda: a.neg_())
    L["k0_noarg"] = bench(lambda: _k0[(1,)](), iters=100)
    L["k1_1tensor"] = bench(lambda: _k1[(1,)](a))
    L["k2_tensor_int"] = bench(lambda: _k2[(1,)](a, 1024))
    L["k3_tensor_int_ce"] = bench(lambda: _k3[(1,)](a, 1024, BLOCK=1024))
    L["k4_tensor_int_2ce"] = bench(lambda: _k4[(1,)](a, 1024, BLOCK=1024, EVEN=True))
    L["seed_run"] = bench(lambda: cand.run(a))
    L["seed_neg_flat_1argish"] = bench(
        lambda: cand._neg_flat[(1,)](a, 1024, BLOCK=1024, EVEN=True))
    L["two_launches"] = bench(
        lambda: (cand._neg_flat[(1,)](a, 1024, BLOCK=1024, EVEN=True),
                 cand._neg_flat[(1,)](a, 1024, BLOCK=1024, EVEN=True)))
    L["eight_launches"] = bench(
        lambda: [cand._neg_flat[(1,)](a, 1024, BLOCK=1024, EVEN=True) for _ in range(8)],
        iters=100)
    # torch floor for an in-place elementwise op
    L["torch_zero_"] = bench(lambda: a.zero_())

    # ---------- precompiled runner (skip JITFunction.run) -------------------
    try:
        ck = _k1.warmup(a, grid=(1,))
        L["compiled_type"] = type(ck).__name__
        L["compiled_attrs"] = [x for x in dir(ck) if not x.startswith("_")]
        try:
            L["compiled_run_sig"] = str(inspect.signature(ck.run))
        except Exception:
            L["compiled_run_sig"] = "unavailable"
        try:
            runner = ck[(1, 1, 1)]
            L["compiled_getitem_1tensor"] = bench(lambda: runner(a))
        except Exception:
            L["compiled_getitem_err"] = traceback.format_exc()[-800:]
        from triton.runtime import driver as tdrv
        stream = tdrv.active.get_current_stream(torch.npu.current_device())
        L["stream_type"] = type(stream).__name__
        meta = getattr(ck, "packed_metadata", None)
        L["packed_metadata"] = repr(meta)[:200]
        try:
            fn = ck.run
            L["compiled_raw_run"] = bench(
                lambda: fn(1, 1, 1, stream, ck.function, meta, None, None, None, a))
        except Exception:
            L["compiled_raw_err"] = traceback.format_exc()[-800:]
        try:
            ckneg = cand._neg_flat.warmup(a, 1024, BLOCK=1024, EVEN=True, grid=(1,))
            rneg = ckneg[(1, 1, 1)]
            L["compiled_neg_run"] = bench(lambda: rneg(a, 1024))
        except Exception:
            L["compiled_neg_err"] = traceback.format_exc()[-800:]
    except Exception:
        L["compiled_outer_err"] = traceback.format_exc()[-1500:]

    # ---------- device-side tiling sweep for 4096x4096 ----------------------
    def sweep(name, t, n, block, iters):
        if iters == 1:
            g = (n + block - 1) // block
            f = lambda: _neg_b[(g,)](t, n, BLOCK=block)
        else:
            span = block * iters
            g = (n + span - 1) // span
            step = g * block
            f = lambda: _neg_gs[(g,)](t, n, BLOCK=block, ITERS=iters, STEP=step)
        return {"grid": g, "block": block, "iters": iters,
                "ms": bench(f, iters=60, warmup=10)}

    for tname, t in (("fp32_4096", big32), ("fp16_4096", big16)):
        n = t.numel()
        recs = []
        for block in (2048, 4096, 8192, 16384):
            recs.append(sweep(tname, t, n, block, 1))
        for block, it in ((4096, 4), (8192, 4), (8192, 8), (16384, 4), (16384, 8)):
            recs.append(sweep(tname, t, n, block, it))
        out["sweep"][tname] = recs
        out["sweep"][tname + "_torch"] = bench(lambda: t.neg_(), iters=60, warmup=10)
        out["sweep"][tname + "_seed"] = bench(lambda: cand.run(t), iters=60, warmup=10)


try:
    main()
except Exception:
    out["fatal"] = traceback.format_exc()
finally:
    ART.parent.mkdir(parents=True, exist_ok=True)
    ART.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str)[:9000])
