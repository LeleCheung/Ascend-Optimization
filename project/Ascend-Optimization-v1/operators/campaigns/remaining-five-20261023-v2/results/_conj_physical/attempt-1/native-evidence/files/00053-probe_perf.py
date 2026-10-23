"""Diagnostic walltime split: candidate vs CPU-fallback reference per timing input.

Not an authoritative measurement -- just shows host launch cost vs device time so
the next experiment targets the right layer.
"""
import importlib.util
import json
import os
import time

import torch
import torch_npu  # noqa: F401

MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "main.py")
spec = importlib.util.spec_from_file_location("cand2", MAIN)
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)

DEV = "npu"
OUT = {}

CASES = [
    ("t0", "float32", 1024, 1729),
    ("t1", "float32", 65536, 1730),
    ("t2", "float32", 1048576, 1731),
    ("t3", "complex64", 1024, 1732),
    ("t4", "complex64", 65536, 1733),
    ("t5", "complex64", 1048576, 1734),
]


def build(dtype, n, seed):
    dt = getattr(torch, dtype)
    g = torch.Generator().manual_seed(seed)
    if dt.is_complex:
        x = torch.complex(torch.randn(n, generator=g), torch.randn(n, generator=g)).to(dtype=dt)
    else:
        x = torch.randn(n, generator=g).to(dtype=dt)
    return x.to(DEV)


def wall(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


def device_ms(fn, iters):
    s = torch.npu.Event(enable_timing=True)
    e = torch.npu.Event(enable_timing=True)
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.npu.synchronize()
    return s.elapsed_time(e) / iters


for name, dtype, n, seed in CASES:
    x = build(dtype, n, seed)
    iters = 50 if n <= 65536 else 20
    r = {}
    try:
        r["cand_ms"] = wall(lambda: cand.run(x), iters)
        r["cand_dev_ms"] = device_ms(lambda: cand.run(x), iters)
        r["empty_ms"] = wall(lambda: torch.empty(x.shape, dtype=x.dtype, device=x.device), iters)
    except Exception as exc:
        r["cand_error"] = repr(exc)
    try:
        r["ref_ms"] = wall(lambda: torch.ops.aten._conj_physical(x), max(5, iters // 5))
    except Exception as exc:
        r["ref_error"] = repr(exc)
    if "cand_ms" in r and "ref_ms" in r:
        r["speedup"] = r["ref_ms"] / r["cand_ms"]
    OUT[name] = r

# host-only cost of one launch with the compiled kernel warmed up
x = build("float32", 1024, 1729)
try:
    OUT["host_only"] = {
        "cand_tiny_ms": wall(lambda: cand.run(x), 200),
        "empty_ms": wall(lambda: torch.empty(x.shape, dtype=x.dtype, device=x.device), 200),
    }
except Exception as exc:
    OUT["host_only"] = {"error": repr(exc)}

# is the pre-bound launcher actually in use, and how much does it save?
try:
    xc = build("complex64", 1024, 1732)
    cand.run(xc)
    out_ref = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    blk = cand._block_for(1024, cand._COPY_TINY, cand._COPY_MID, cand._COPY_LARGE)

    def ordinary():
        cand._copy_flat_kernel[(-(-1024 // blk),)](x, out_ref, 1024, BLOCK=blk)

    OUT["launch_paths"] = {
        "copy_cache_entries": len(cand._copy_cache),
        "conj_cache_entries": len(cand._conj_cache),
        "copy_keys": [str(k) for k in cand._copy_cache],
        "ordinary_jit_ms": wall(ordinary, 200),
        "prebound_ms": wall(lambda: cand.run(x), 200),
    }
    # an unaligned view must get its own specialization and stay correct
    base = torch.randn(4097, device=DEV)
    view = base[1:]
    exp = torch.ops.aten._conj_physical(view.cpu())
    got = cand.run(view).cpu()
    OUT["launch_paths"]["unaligned_ok"] = bool(torch.allclose(got, exp, atol=1e-5))
    OUT["launch_paths"]["unaligned_ptr_mod16"] = int(view.data_ptr() & 15)
    OUT["launch_paths"]["copy_entries_after_unaligned"] = len(cand._copy_cache)
    OUT["launch_paths"]["unaligned_ms"] = wall(lambda: cand.run(view), 100)
except Exception as exc:
    OUT["launch_paths"] = {"error": repr(exc)}

art = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
os.makedirs(art, exist_ok=True)
with open(os.path.join(art, "probe_perf.json"), "w") as f:
    json.dump(OUT, f, indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str))
