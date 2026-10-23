"""Probe ATen _conj_physical reference behaviour on the target device.

Records: output strides/layout rules, conj-bit handling, real/complex dispatch,
per-call walltime for the timing workloads, and evidence about CPU fallback.
"""
import json
import os
import time
import traceback

import torch
import torch_npu  # noqa: F401

OUT = {}


def gen(spec, seed):
    dtype = getattr(torch, spec["dtype"])
    g = torch.Generator().manual_seed(seed)
    shape = spec["shape"]
    x = torch.randn(shape, generator=g).to(dtype=dtype, device="npu")
    if dtype.is_complex:
        x = torch.complex(
            torch.randn(shape, generator=g), torch.randn(shape, generator=g)
        ).to(dtype=dtype, device="npu")
    if spec.get("transpose"):
        x = x.transpose(-1, -2)
    if spec.get("conj"):
        x = x.conj()
    return x


CORRECTNESS = [
    ("c0", {"dtype": "float32", "shape": [7, 9], "transpose": True, "conj": True}, 1729),
    ("c1", {"dtype": "float32", "shape": [7, 9], "transpose": True, "conj": False}, 1730),
    ("c2", {"dtype": "float32", "shape": [7, 9], "transpose": False, "conj": True}, 1731),
    ("c3", {"dtype": "float32", "shape": [7, 9], "transpose": False, "conj": False}, 1732),
    ("c4", {"dtype": "complex64", "shape": [7, 9], "transpose": True, "conj": True}, 1733),
    ("c5", {"dtype": "complex64", "shape": [7, 9], "transpose": True, "conj": False}, 1734),
    ("c6", {"dtype": "complex64", "shape": [7, 9], "transpose": False, "conj": True}, 1735),
    ("c7", {"dtype": "complex64", "shape": [7, 9], "transpose": False, "conj": False}, 1736),
]

TIMING = [
    ("t0", {"dtype": "float32", "shape": [1024]}, 1729),
    ("t1", {"dtype": "float32", "shape": [65536]}, 1730),
    ("t2", {"dtype": "float32", "shape": [1048576]}, 1731),
    ("t3", {"dtype": "complex64", "shape": [1024]}, 1732),
    ("t4", {"dtype": "complex64", "shape": [65536]}, 1733),
    ("t5", {"dtype": "complex64", "shape": [1048576]}, 1734),
]


def timeit(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


# ---- correctness specs: layout / bit facts ---------------------------------
layout = {}
for name, spec, seed in CORRECTNESS:
    try:
        x = gen(spec, seed)
        ref = torch.ops.aten._conj_physical(x)
        cpu_x = x.cpu()
        logical = torch.conj(cpu_x) if x.is_conj() else cpu_x
        entry = {
            "in_dtype": str(x.dtype),
            "in_stride": list(x.stride()),
            "in_is_conj": bool(x.is_conj()),
            "in_is_contig": bool(x.is_contiguous()),
            "out_stride": list(ref.stride()),
            "out_is_contig": bool(ref.is_contiguous()),
            "out_is_conj": bool(ref.is_conj()),
            "out_dtype": str(ref.dtype),
            "out_shape": list(ref.shape),
            "aliases_input": bool(ref.data_ptr() == x.data_ptr()),
            "matches_conj_logical": bool(
                torch.equal(ref.cpu(), torch.conj(logical))
            ),
            "out_equals_conj_physical_ref": bool(
                torch.allclose(
                    ref.cpu(),
                    torch.ops.aten._conj_physical(logical),
                    rtol=1e-5,
                    atol=1e-6,
                )
            ),
        }
        layout[name] = entry
    except Exception:
        layout[name] = {"error": traceback.format_exc()}

OUT["layout"] = layout

# ---- extra facts -----------------------------------------------------------
facts = {}
try:
    r = torch.randn(3, 4, device="npu")
    facts["real_conj_is_conj"] = bool(r.conj().is_conj())
    facts["real_conj_aliases"] = bool(r.conj().data_ptr() == r.data_ptr())
    facts["real_conj_returns_self"] = bool(r.conj() is r)
except Exception:
    facts["real_conj_error"] = traceback.format_exc()

try:
    c = torch.randn(3, 4, dtype=torch.complex64, device="npu")
    cc = c.conj()
    facts["complex_conj_is_conj"] = bool(cc.is_conj())
    facts["complex_conj_aliases"] = bool(cc.data_ptr() == c.data_ptr())
    raw = cc.conj() if cc.is_conj() else cc
    v = raw.view_as_real()
    facts["view_as_real_ok"] = True
    facts["view_as_real_shape"] = list(v.shape)
    facts["view_as_real_stride"] = list(v.stride())
    facts["view_as_real_ptr_same"] = bool(v.data_ptr() == c.data_ptr())
    t = c.transpose(-1, -2).conj()
    rawt = t.conj() if t.is_conj() else t
    vt = rawt.view_as_real()
    facts["complex_transposed_view_as_real_shape"] = list(vt.shape)
    facts["complex_transposed_view_as_real_stride"] = list(vt.stride())
except Exception:
    facts["complex_probe_error"] = traceback.format_exc()

try:
    facts["empty_like_transposed_complex_stride"] = list(
        torch.empty_like(
            torch.randn(7, 9, dtype=torch.complex64, device="npu").transpose(-1, -2)
        ).stride()
    )
    facts["empty_like_transposed_real_stride"] = list(
        torch.empty_like(torch.randn(7, 9, device="npu").transpose(-1, -2)).stride()
    )
    # gap slice (non dense): empty_like should fall back to contiguous
    base = torch.randn(7, 18, device="npu")
    facts["empty_like_gap_slice_stride"] = list(torch.empty_like(base[:, ::2]).stride())
    facts["ref_gap_slice_stride"] = list(
        torch.ops.aten._conj_physical(base[:, ::2]).stride()
    )
except Exception:
    facts["empty_like_error"] = traceback.format_exc()
OUT["facts"] = facts

# ---- timing of the reference ----------------------------------------------
times = {}
for name, spec, seed in TIMING:
    try:
        x = gen(spec, seed)
        fn = lambda: torch.ops.aten._conj_physical(x)
        iters = 50 if x.numel() <= 65536 else 20
        ms = timeit(fn, iters)
        # crude breakdown: empty_like alone (allocation cost)
        ms_alloc = timeit(lambda: torch.empty_like(x), iters)
        out = fn()
        times[name] = {
            "dtype": str(x.dtype),
            "numel": x.numel(),
            "ref_ms": ms,
            "empty_like_ms": ms_alloc,
            "out_device": str(out.device),
            "ref_impl_ratio_vs_bytes": None,
        }
    except Exception:
        times[name] = {"error": traceback.format_exc()}
OUT["times"] = times

# ---- does complex reference really run on NPU? ----------------------------
try:
    from torch.profiler import ProfilerActivity, profile

    x = torch.randn(1 << 20, dtype=torch.complex64, device="npu")
    x = torch.complex(x.real, x.imag).to(torch.complex64)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
        torch.ops.aten._conj_physical(x)
        torch.npu.synchronize()
    evts = []
    for e in prof.events():
        if e.device_type is not None and str(e.device_type).lower().endswith("npu"):
            evts.append({"name": e.name, "dur": getattr(e, "device_time", None)})
    OUT["complex_profile"] = evts[:40]
    OUT["complex_profile_all_names"] = [e.name for e in prof.events()][:60]
except Exception:
    OUT["complex_profile_error"] = traceback.format_exc()

art = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
os.makedirs(art, exist_ok=True)
with open(os.path.join(art, "probe_ref.json"), "w") as f:
    json.dump(OUT, f, indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str)[:12000])
