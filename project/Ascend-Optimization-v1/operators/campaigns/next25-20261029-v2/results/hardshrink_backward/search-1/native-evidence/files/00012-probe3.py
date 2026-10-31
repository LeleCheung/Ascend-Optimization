"""Bounded decomposition of the per-call floor of the round-2 candidate (<=120 s).

Measures, on target, with the real candidate module:
  * full run() for representative timing shapes
  * run() with the dispatch replaced by a no-op (allocation + Python checks only)
  * empty_like alone, both with the result discarded and with it kept alive
  * micro-costs of every metadata read used in the hot path
  * the prebuilt launcher closure with a preallocated output (launch + kernel only)
  * a check-free minimal function (empty_like + closure) = achievable floor
  * the ATen reference in the same process
"""
import json
import os
import sys
import time
import traceback

import torch
import torch_npu
import triton

sys.path.insert(0, os.getcwd())
ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
os.makedirs(ART, exist_ok=True)
OUT = {"cases": {}, "micro": {}, "notes": []}
T0 = time.time()
_KEEP = None


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

SHAPES = [
    ("f16_1k", 1024, torch.float16, 1729),
    ("f16_64k", 65536, torch.float16, 1730),
    ("f16_1m", 1048576, torch.float16, 1731),
    ("f16_4k2", 4096 * 4096, torch.float16, 1732),
    ("f32_1k", 1024, torch.float32, 1737),
    ("f32_4k2", 4096 * 4096, torch.float32, 1740),
]

for name, n, dt, seed in SHAPES:
    g = torch.Generator().manual_seed(seed)
    go = torch.randn(n, generator=g).to(dt).npu()
    x = torch.randn(n, generator=g).to(dt).npu()
    it = 200 if n <= 65536 else 60
    c = {}
    try:
        c["aten_ms"] = bench(lambda: REF(go, x, 0.5), iters=it)
        c["full_ms"] = bench(lambda: m.run(go, x, 0.5), iters=it)
        # allocation + python checks only
        key = (n, dt, go.get_device())
        saved = m._CACHE.get(key)
        m._CACHE[key] = lambda gg, xx, oo, ll: None
        c["no_dispatch_ms"] = bench(lambda: m.run(go, x, 0.5), iters=it)
        if saved is not None:
            m._CACHE[key] = saved
        c["empty_like_ms"] = bench(lambda: torch.empty_like(go), iters=it)
        # launch + kernel only, preallocated output
        out = torch.empty_like(go)
        call = m._CACHE.get(key)
        if call is not None:
            c["launch_kernel_ms"] = bench(lambda: call(go, x, out, 0.5), iters=it)

            def minimal():
                o = torch.empty_like(go)
                call(go, x, o, 0.5)
                return o

            c["minimal_ms"] = bench(minimal, iters=it)

            def minimal_keep():
                global _KEEP
                o = torch.empty_like(go)
                _KEEP = o
                call(go, x, o, 0.5)
                return o

            c["minimal_keep_ms"] = bench(minimal_keep, iters=it)
    except Exception:
        c["err"] = traceback.format_exc()[-900:]
    OUT["cases"][name] = c
    note(f"{name} done")

# ---------------------------------------------------------------- micro costs
go, x = torch.randn(1024, dtype=torch.float16).npu(), torch.randn(1024).half().npu()
key = (1024, torch.float16, go.get_device())
call = m._CACHE.get(key)
OUT["micro"]["dtype"] = bench(lambda: go.dtype, iters=2000)
OUT["micro"]["numel"] = bench(lambda: go.numel(), iters=2000)
OUT["micro"]["is_contiguous"] = bench(lambda: go.is_contiguous(), iters=2000)
OUT["micro"]["get_device"] = bench(lambda: go.get_device(), iters=2000)
OUT["micro"]["shape_eq"] = bench(lambda: x.shape != go.shape, iters=2000)
OUT["micro"]["float_05"] = bench(lambda: float(0.5), iters=2000)
OUT["micro"]["tuple_key"] = bench(
    lambda: m._CACHE.get((go.numel(), go.dtype, go.get_device())), iters=2000
)
OUT["micro"]["data_ptr"] = bench(lambda: go.data_ptr(), iters=2000)
if call is not None:
    OUT["micro"]["closure_prealloc"] = bench(lambda: call(go, x, go, 0.5), iters=2000)
note("micro done")

with open(os.path.join(ART, "probe3.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print(json.dumps(OUT, indent=1)[:9000])
