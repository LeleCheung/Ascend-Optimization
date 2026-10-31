"""Bounded floor calibration + fast-path prototype for hardshrink_backward (<=150 s).

Calibrates what a *single* allocate+launch ATen op costs on this target, and
measures prototypes of a leaner run() hot path against the current module.
"""
import json
import os
import sys
import time
import traceback

import torch
import torch_npu

sys.path.insert(0, os.getcwd())
ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
os.makedirs(ART, exist_ok=True)
OUT = {"cases": {}, "floor": {}, "notes": []}
T0 = time.time()


def note(m):
    OUT["notes"].append(f"{time.time() - T0:7.2f}s {m}")


def bench(fn, iters=200, warm=20):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


import importlib

m = importlib.import_module("tmp.main")
REF = torch.ops.aten.hardshrink_backward
_CACHE2 = {}


def build2(grad_out, out, n):
    """Same compiled launcher as the module, but resolved for the prototype."""
    esize = out.element_size()
    cap = m._CAP_4B if esize > 2 else m._CAP_2B
    if n % cap == 0:
        jitfn, block, grid, extra = m._hsb_flat, cap, n // cap, ()
    else:
        jitfn, block, extra = m._hsb_masked, m._BASE, (n,)
        grid = -(-n // m._BASE)
    stream_of = m._stream_getter(grad_out.get_device())
    if extra:
        ck = jitfn.warmup(out, out, out, extra[0], 0.5, BLOCK=block,
                          num_warps=m._WARPS, grid=(1,))
    else:
        ck = jitfn.warmup(out, out, out, 0.5, BLOCK=block, num_warps=m._WARPS, grid=(1,))
    L, fn_, pm = ck.run.launch, ck.function, ck.packed_metadata
    if extra:
        n0 = extra[0]

        def call(g, x, o, lam, _L=L, _f=fn_, _p=pm, _g=grid, _s=stream_of, _n=n0, _b=block):
            _L(_g, 1, 1, _s(), _f, _p, None, None, None,
               g.data_ptr(), x.data_ptr(), o.data_ptr(), _n, lam, _b)
    else:

        def call(g, x, o, lam, _L=L, _f=fn_, _p=pm, _g=grid, _s=stream_of, _b=block):
            _L(_g, 1, 1, _s(), _f, _p, None, None, None,
               g.data_ptr(), x.data_ptr(), o.data_ptr(), lam, _b)
    return call


def run_v2(grad_out, self_t, lambd=0.5):
    """Lean hot path: one allocation, three cheap metadata reads, one dict hit."""
    out = torch.empty_like(grad_out)
    if grad_out.is_contiguous():
        n = grad_out.numel()
        key = (n, grad_out.dtype, grad_out.get_device())
        call = _CACHE2.get(key)
        if call is None:
            call = build2(grad_out, out, n)
            _CACHE2[key] = call
        call(grad_out, self_t, out, float(lambd))
        return out
    return m.hardshrink_backward(grad_out, self_t, lambd)


def run_v3(grad_out, self_t, lambd=0.5):
    """v2 but the pointers are read in run() instead of inside the closure."""
    out = torch.empty_like(grad_out)
    if grad_out.is_contiguous():
        n = grad_out.numel()
        ent = _CACHE2.get(("p", n, grad_out.dtype, grad_out.get_device()))
        if ent is None:
            call = build2(grad_out, out, n)
            ent = (call, m._stream_getter(grad_out.get_device()))
            _CACHE2[("p", n, grad_out.dtype, grad_out.get_device())] = ent
        ent[0](grad_out, self_t, out, float(lambd))
        return out
    return m.hardshrink_backward(grad_out, self_t, lambd)


SHAPES = [
    ("f16_1k", 1024, torch.float16, 1729),
    ("f16_1m", 1048576, torch.float16, 1731),
    ("f32_1k", 1024, torch.float32, 1737),
]

for name, n, dt, seed in SHAPES:
    g = torch.Generator().manual_seed(seed)
    go = torch.randn(n, generator=g).to(dt).npu()
    x = torch.randn(n, generator=g).to(dt).npu()
    c = {}
    try:
        c["aten_ms"] = bench(lambda: REF(go, x, 0.5))
        c["v1_full_ms"] = bench(lambda: m.run(go, x, 0.5))
        c["v2_ms"] = bench(lambda: run_v2(go, x, 0.5))
        c["v3_ms"] = bench(lambda: run_v3(go, x, 0.5))
        ref = REF(go, x, 0.5)
        c["v2_ok"] = bool(torch.equal(run_v2(go, x, 0.5), ref))
        c["v3_ok"] = bool(torch.equal(run_v3(go, x, 0.5), ref))
        # floor calibration: single allocate + launch ATen ops
        c["aten_mul1"] = bench(lambda: torch.mul(go, 1.0))
        c["aten_clone"] = bench(lambda: go.clone())
        c["aten_relu"] = bench(lambda: torch.relu(go))
        c["aten_neg"] = bench(lambda: torch.neg(go))
        c["aten_zeros_like"] = bench(lambda: torch.zeros_like(go))
        c["empty_like"] = bench(lambda: torch.empty_like(go))
    except Exception:
        c["err"] = traceback.format_exc()[-900:]
    OUT["cases"][name] = c
    note(f"{name} done")

with open(os.path.join(ART, "probe4.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print(json.dumps(OUT, indent=1)[:9000])
