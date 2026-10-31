"""Bounded probe: launcher per-argument cost and num_stages for the 16.7M case (<=150 s)."""
import json
import os
import sys
import time
import traceback

import torch
import torch_npu
import triton
import triton.language as tl

sys.path.insert(0, os.getcwd())
ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
os.makedirs(ART, exist_ok=True)
OUT = {"launch": {}, "stages": {}, "notes": []}
T0 = time.time()


def note(m):
    OUT["notes"].append(f"{time.time() - T0:7.2f}s {m}")


def bench(fn, iters=300, warm=30):
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
dev = torch.npu.current_device()
rawf = torch_npu._C._npu_getCurrentRawStreamNoWait


@triton.jit
def _d1(p0, BLOCK: tl.constexpr):
    tl.store(p0 + tl.program_id(0), 0.0)


@triton.jit
def _d4(p0, p1, p2, f, BLOCK: tl.constexpr):
    tl.store(p0 + tl.program_id(0), f)


g = torch.randn(1024, dtype=torch.float16).npu()
x = torch.randn(1024, dtype=torch.float16).npu()
o = torch.empty_like(g)

for tag, jitfn, tail in (("d1", _d1, (1024,)), ("d4", _d4, (0.5, 1024))):
    try:
        ck = jitfn.warmup(*( [o] if tag == "d1" else [g, x, o, 0.5] ),
                          BLOCK=1024, num_warps=4, grid=(1,))
        L, f_, pm = ck.run.launch, ck.function, ck.packed_metadata
        if tag == "d1":
            def call(_L=L, _f=f_, _p=pm):
                _L(1, 1, 1, rawf(dev), _f, _p, None, None, None, o.data_ptr(), 1024)
        else:
            def call(_L=L, _f=f_, _p=pm):
                _L(1, 1, 1, rawf(dev), _f, _p, None, None, None,
                   g.data_ptr(), x.data_ptr(), o.data_ptr(), 0.5, 1024)
        call()
        torch.npu.synchronize()
        OUT["launch"][tag + "_ms"] = bench(call)
    except Exception:
        OUT["launch"][tag + "_err"] = traceback.format_exc()[-500:]
    note(f"{tag} done")

# num_stages sweep on the 16.7M float16 case, BLOCK cap
go = torch.randn(4096 * 4096, 1, dtype=torch.float16).npu().view(-1)
xx = torch.randn(4096 * 4096, 1, dtype=torch.float16).npu().view(-1)
out = torch.empty_like(go)
ref = REF(go, xx, 0.5)
OUT["stages"]["aten_ms"] = bench(lambda: REF(go, xx, 0.5), iters=60)
block = m._CAP_2B
grid = go.numel() // block
for stages in (1, 2, 3):
    try:
        ck = m._hsb_flat.warmup(go, xx, out, 0.5, BLOCK=block,
                                num_warps=m._WARPS, num_stages=stages, grid=(1,))
        L, f_, pm = ck.run.launch, ck.function, ck.packed_metadata

        def call(_L=L, _f=f_, _p=pm, _g=grid, _b=block):
            _L(_g, 1, 1, rawf(dev), _f, _p, None, None, None,
               go.data_ptr(), xx.data_ptr(), out.data_ptr(), 0.5, _b)

        call()
        torch.npu.synchronize()
        OUT["stages"][f"s{stages}_bitmatch"] = bool(torch.equal(out, ref))
        OUT["stages"][f"s{stages}_ms"] = bench(call, iters=60)
    except Exception:
        OUT["stages"][f"s{stages}_err"] = traceback.format_exc()[-500:]
    note(f"stages {stages} done")

with open(os.path.join(ART, "probe5.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print(json.dumps(OUT, indent=1)[:6000])
