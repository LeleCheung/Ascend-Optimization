"""Verify the pre-bound launch fallback: a raising launcher must still yield
correct output via the ordinary dispatch, and must not change the measured floor.
"""
import importlib.util
import json
import os
import time

import torch
import torch_npu  # noqa: F401

MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "main.py")
spec = importlib.util.spec_from_file_location("cand4", MAIN)
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)

DEV = "npu"
OUT = {}


def wall(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


x = torch.randn(4096, device=DEV)
exp = torch.ops.aten._conj_physical(x.cpu())
cand.run(x)
blk = cand._block_for(4096, cand._COPY_TINY, cand._COPY_MID, cand._COPY_LARGE)
key = (torch.float32, blk, x.data_ptr() & 15, 0)
entry = cand._copy_cache[key]


def boom(*a, **k):
    raise RuntimeError("simulated launcher rejection")


cand._copy_cache[key] = (boom, entry[1], entry[2])
out = cand.run(x)
OUT["fallback"] = {
    "values_ok": bool(torch.equal(out.cpu(), exp)),
    "out_contiguous": bool(out.is_contiguous()),
}
cand._copy_cache[key] = entry

# complex variant
xc = torch.complex(torch.randn(4096), torch.randn(4096)).to(device=DEV)
expc = torch.ops.aten._conj_physical(xc.cpu())
oc = cand.run(xc)
blkc = cand._block_for(2 * xc.numel(), cand._CONJ_TINY, cand._CONJ_MID, cand._CONJ_LARGE)
keyc = (torch.complex64, blkc, True, xc.data_ptr() & 15, oc.data_ptr() & 15)
entryc = cand._conj_cache.get(keyc)
if entryc is not None:
    cand._conj_cache[keyc] = (boom, entryc[1], entryc[2])
    oc2 = cand.run(xc)
    OUT["fallback_complex"] = {
        "values_ok": bool(torch.allclose(oc2.cpu(), expc, rtol=1.5e-2, atol=1.5e-2)),
        "key_found": True,
    }
    cand._conj_cache[keyc] = entryc
else:
    OUT["fallback_complex"] = {"key_found": False, "keys": [str(k) for k in cand._conj_cache]}

OUT["floor_ms"] = {
    "run_f32_1024": wall(lambda: cand.run(torch.randn(1024, device=DEV) if False else x), 2000),
    "run_c64_1024": wall(lambda: cand.run(xc), 1000),
}

art = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
os.makedirs(art, exist_ok=True)
with open(os.path.join(art, "probe_fallback.json"), "w") as f:
    json.dump(OUT, f, indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str))
