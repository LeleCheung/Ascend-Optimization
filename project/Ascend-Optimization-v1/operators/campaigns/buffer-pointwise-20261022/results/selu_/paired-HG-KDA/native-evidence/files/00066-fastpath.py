"""Locate why the candidate's direct C-launcher fast path degrades, and cost it.

Also measures queue-throughput (host) vs device for the large shapes.
"""
import importlib.util
import json
import os
import time
import traceback

import torch
import torch_npu
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
out = {}


def save():
    with open(os.path.join(ART, "fastpath.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)


def timeit_host(fn, iters=300, warm=30):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    t2 = time.perf_counter()
    return {"host_us": (t1 - t0) / iters * 1e6, "total_us": (t2 - t0) / iters * 1e6}


spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)

x = torch.randn(1024, dtype=torch.float16, device="npu")
n = x.numel()
block = 1024

# ------------------------------------------------- step-by-step fast path replay
steps = {}


def rec(name, fn):
    try:
        steps[name] = {"ok": True, "value": repr(fn())[:200]}
        return fn
    except Exception:
        steps[name] = {"ok": False, "traceback": traceback.format_exc()[-1500:]}
        return None


compiled = None
try:
    compiled = cand._selu_nomask.warmup(x, BLOCK=block, grid=(1,))
    steps["warmup"] = {"ok": True, "type": type(compiled).__name__}
except Exception:
    steps["warmup"] = {"ok": False, "traceback": traceback.format_exc()[-1500:]}
save()

if compiled is not None:
    launcher = compiled.run
    steps["launcher_type"] = {"ok": True, "value": type(launcher).__name__}
    for attr in ("compile_only", "enable_msprof_register_tensor"):
        steps["getattr_" + attr] = {
            "ok": True,
            "value": repr(getattr(launcher, attr, "MISSING")),
        }
    steps["metadata_debug_enabled"] = {
        "ok": True,
        "value": repr(getattr(launcher.metadata, "debug_enabled", "MISSING")),
    }
    steps["metadata_keys"] = {
        "ok": True,
        "value": repr(sorted(vars(launcher.metadata).keys()))[:400]
        if hasattr(launcher.metadata, "__dict__")
        else repr(type(launcher.metadata)),
    }
    for expr, fn in [
        ("call_attr", lambda: launcher.launch),
        ("function_attr", lambda: compiled.function),
        ("packed_metadata_attr", lambda: compiled.packed_metadata),
        ("resolve_stream", lambda: cand._resolve_stream(x.device.index)),
        ("torch_npu_raw_stream_attr", lambda: torch_npu._C._npu_getCurrentRawStreamNoWait),
        ("driver_get_current_stream", lambda: __import__("triton.runtime.driver", fromlist=["driver"]).driver.active.get_current_stream(x.device.index)),
    ]:
        rec(expr, fn)
    save()

# ------------------------------------------------------------------- launch cos
if compiled is not None:
    launcher = compiled.run
    call = launcher.launch
    function = compiled.function
    metadata = compiled.packed_metadata
    try:
        stream = cand._resolve_stream(x.device.index)
    except Exception:
        stream = None
    address_type = cand._hgkda_address_type(n)
    ptr = x.data_ptr()
    grid = 1

    variants = {}

    def add(name, fn):
        try:
            fn()
            torch.npu.synchronize()
            variants[name] = timeit_host(fn)
        except Exception:
            variants[name] = {"error": traceback.format_exc()[-600:]}

    if stream is not None:
        s = stream()
        add(
            "A_plain_int",
            lambda: call(grid, 1, 1, s, function, metadata, None, None, None, ptr, None),
        )
        add(
            "B_int_subclass",
            lambda: call(grid, 1, 1, s, function, metadata, None, None, None, address_type(ptr), None),
        )
        add(
            "C_stream_call_each_time",
            lambda: call(grid, 1, 1, stream(), function, metadata, None, None, None, ptr, None),
        )
        # a fresh int() per call like a robust impl would do
        add(
            "D_int_wrap",
            lambda: call(grid, 1, 1, s, function, metadata, None, None, None, int(ptr), None),
        )
    out["launch_variants"] = variants
    save()

    # full candidate run vs reference-style wrapper
    def ref_style(A):
        kwargs = locals().copy()
        return {"selu_": torch.selu_}[kwargs["A"] and "selu_"](kwargs["A"])

    out["run_vs_ref"] = {
        "candidate_run": timeit_host(lambda: cand.run(x)),
        "torch_selu_direct": timeit_host(lambda: torch.selu_(x)),
        "torch_selu_refstyle": timeit_host(lambda: ref_style(x)),
    }
    save()

del x

# ------------------------------------------- large shapes: host floor and total
for tag, shape, dtype, blk in [
    ("bf16_1M", (1048576,), torch.bfloat16, 8192),
    ("fp16_4096x4096", (4096, 4096), torch.float16, 16384),
    ("fp32_4096x4096", (4096, 4096), torch.float32, 8192),
]:
    t = torch.randn(shape, dtype=dtype, device="npu")
    nn_ = t.numel()
    p = t.data_ptr()
    try:
        comp = cand._selu_nomask.warmup(t, BLOCK=blk, grid=(1,))
        cc = comp.run.launch
        ff = comp.function
        mm = comp.packed_metadata
        ss = cand._resolve_stream(t.device.index)()
        gg = nn_ // blk
        entry = {
            "grid": gg,
            "candidate_run": timeit_host(lambda: cand.run(t), iters=100, warm=20),
            "floor_plain_int": timeit_host(lambda: cc(gg, 1, 1, ss, ff, mm, None, None, None, p, None), iters=100, warm=20),
            "torch_ref": timeit_host(lambda: torch.selu_(t), iters=100, warm=20),
        }
        out[f"large_{tag}"] = entry
        print(tag, entry, flush=True)
    except Exception:
        out[f"large_{tag}"] = {"error": traceback.format_exc()[-800:]}
    del t
    save()

print(json.dumps(out, indent=2, default=str), flush=True)
