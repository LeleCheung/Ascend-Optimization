"""Bounded host-path diagnostic for selu_ candidate: paired host cost breakdown.

Measures, on the real target, per-call host cost of several launch variants for
the same in-place SELU, so we can tell whether the walltime latency on small
workloads is host-launch bound and which Python-side work is worth removing.
Every launch uses the real tensor address; no dummy/null pointers.
"""
import importlib.util
import json
import os
import time

import torch
import torch_npu
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
out = {}


def stage(name):
    print(f"[stage] {name}", flush=True)


def save():
    with open(os.path.join(ART, "host_cost.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)


stage("import candidate")
spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)

dev = torch.npu.current_device()
out["torch"] = torch.__version__
out["torch_npu"] = getattr(torch_npu, "__version__", "?")
out["triton"] = triton.__version__


def timeit_host(fn, iters=400, warm=50):
    """Host-throughput: queue many launches, sync once."""
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    t2 = time.perf_counter()
    return {
        "host_us": (t1 - t0) / iters * 1e6,
        "total_us": (t2 - t0) / iters * 1e6,
    }


def timeit_sync(fn, iters=200, warm=20):
    """Per-call sync: host launch + device task + sync latency."""
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
        torch.npu.synchronize()
    t1 = time.perf_counter()
    return {"per_call_us": (t1 - t0) / iters * 1e6}


# ---------------------------------------------------------------- inspect path
stage("inspect launch path")
probe = torch.randn(1024, dtype=torch.float16, device="npu")
cand.run(probe)
key = (probe.numel(), probe.data_ptr() & 15)
entry = cand._FAST.get(key)
info = {"fast_entry_present": entry is not None}
if entry is not None:
    fn = entry[1]
    code = getattr(fn, "__code__", None)
    info["launch_freevars"] = list(code.co_freevars) if code is not None else None
    info["is_direct_c_launcher"] = "call" in (code.co_freevars if code else ())
try:
    compiled = cand._selu_nomask.warmup(probe, BLOCK=1024, grid=(1,))
    info["compiled_run_type"] = type(compiled.run).__name__
    info["has_launch_attr"] = hasattr(compiled.run, "launch")
    info["enable_msprof_register_tensor"] = getattr(
        compiled.run, "enable_msprof_register_tensor", "MISSING"
    )
    info["compile_only"] = getattr(compiled.run, "compile_only", "MISSING")
    info["debug_enabled"] = getattr(compiled.run.metadata, "debug_enabled", "MISSING")
    info["launch_attr_type"] = type(getattr(compiled.run, "launch", None)).__name__
except Exception as exc:  # noqa: BLE001
    info["compiled_exc"] = repr(exc)
out["path"] = info
save()

# ------------------------------------------------------- direct launcher floor
stage("build direct launcher floor")


def build_floor(compiled, n, block, masked):
    call = compiled.run.launch
    function = compiled.function
    metadata = compiled.packed_metadata
    stream = cand._resolve_stream(probe.device.index)
    grid = triton.cdiv(n, block) if masked else n // block
    address = cand._hgkda_address_type(n)
    if masked:

        def floor_int(ptr, grid=grid, call=call, stream=stream, f=function, m=metadata, n=n):
            call(grid, 1, 1, stream(), f, m, None, None, None, ptr, n, None)

        def floor_sub(ptr, grid=grid, call=call, stream=stream, f=function, m=metadata, n=n, addr=address):
            call(grid, 1, 1, stream(), f, m, None, None, None, addr(ptr), n, None)
    else:

        def floor_int(ptr, grid=grid, call=call, stream=stream, f=function, m=metadata):
            call(grid, 1, 1, stream(), f, m, None, None, None, ptr, None)

        def floor_sub(ptr, grid=grid, call=call, stream=stream, f=function, m=metadata, addr=address):
            call(grid, 1, 1, stream(), f, m, None, None, None, addr(ptr), None)
    return floor_int, floor_sub


n_small = 1024
x_small = torch.randn(n_small, dtype=torch.float16, device="npu")
compiled_small = cand._selu_nomask.warmup(x_small, BLOCK=1024, grid=(1,))
floor_int, floor_sub = build_floor(compiled_small, n_small, 1024, False)

# ------------------------------------------------------------- micro variants
numel = torch.Tensor.numel
data_ptr = torch.Tensor.data_ptr
is_contig = torch.Tensor.is_contiguous
_FAST = cand._FAST
FAST_GET = _FAST.get
_torch_selu_ = torch.selu_


def v_current(A=x_small):
    return cand.run(A)


def v_prebound(A=x_small):
    """Bound-method lookups hoisted to module scope."""
    n = numel(A)
    ptr = data_ptr(A)
    entry = FAST_GET((n, ptr & 15))
    if entry is not None and A.dtype is entry[0] and is_contig(A):
        entry[1](A, ptr)
        return A
    return cand._dispatch(A, n, ptr)


def v_torch(A=x_small):
    return _torch_selu_(A)


_small_ptr = x_small.data_ptr()


def v_floor_sub():
    floor_sub(_small_ptr)


def v_floor_int():
    floor_int(_small_ptr)


stage("bench small fp16 host-throughput")
bench = {}
for name, fn in [
    ("current_run", v_current),
    ("prebound_methods", v_prebound),
    ("floor_int_subclass", v_floor_sub),
    ("floor_plain_int", v_floor_int),
    ("torch_selu_ref", v_torch),
]:
    try:
        bench[name] = timeit_host(fn)
    except Exception as exc:  # noqa: BLE001
        bench[name] = {"error": repr(exc)}
    print(f"  {name}: {bench[name]}", flush=True)
out["bench_fp16_1024_host"] = bench
save()

stage("bench small fp16 per-call sync")
sync_bench = {}
for name, fn in [
    ("current_run", v_current),
    ("floor_plain_int", v_floor_int),
    ("torch_selu_ref", v_torch),
]:
    try:
        sync_bench[name] = timeit_sync(fn)
    except Exception as exc:  # noqa: BLE001
        sync_bench[name] = {"error": repr(exc)}
    print(f"  {name}: {sync_bench[name]}", flush=True)
out["bench_fp16_1024_sync"] = sync_bench
save()
del probe, x_small

# ------------------------------------------------- medium/large: host vs dev
for tag, shape, dtype, block in [
    ("bf16_1M", (1048576,), torch.bfloat16, 8192),
    ("fp32_4096x4096", (4096, 4096), torch.float32, 8192),
    ("fp16_4096x4096", (4096, 4096), torch.float16, 16384),
]:
    stage(f"bench {tag}")
    t = torch.randn(shape, dtype=dtype, device="npu")
    try:
        n = t.numel()
        prog = cand._selu_nomask.warmup(t, BLOCK=block, grid=(1,))
        f_int, _ = build_floor(prog, n, block, False)
        _ptr = t.data_ptr()
        out[f"bench_{tag}"] = {
            "candidate_host": timeit_host(lambda A=t: cand.run(A), iters=200, warm=20),
            "floor_plain_int_host": timeit_host(lambda f=f_int, p=_ptr: f(p), iters=200, warm=20),
            "torch_ref_host": timeit_host(lambda A=t: _torch_selu_(A), iters=200, warm=20),
            "candidate_sync": timeit_sync(lambda A=t: cand.run(A), iters=50, warm=10),
            "torch_ref_sync": timeit_sync(lambda A=t: _torch_selu_(A), iters=50, warm=10),
        }
        print(f"  {out['bench_' + tag]}", flush=True)
    except Exception as exc:  # noqa: BLE001
        out[f"bench_{tag}"] = {"error": repr(exc)}
        print(f"  error {exc!r}", flush=True)
    del t
    save()

save()
print(json.dumps(out, indent=2, default=str), flush=True)
