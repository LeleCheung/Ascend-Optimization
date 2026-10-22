"""Component-level host cost of the per-call path, plus the fast-path fallback cause."""
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
    with open(os.path.join(ART, "components.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)


spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)

x = torch.randn(1024, dtype=torch.float16, device="npu")
n = x.numel()
ptr = x.data_ptr()
block = 1024

# ------------------------------------------------------------- fallback replay
steps = {}


def rec(name, fn):
    try:
        steps[name] = {"ok": True, "value": repr(fn())[:300]}
        return True
    except Exception:
        steps[name] = {"ok": False, "traceback": traceback.format_exc()[-1200:]}
        return False


compiled = None
try:
    compiled = cand._selu_nomask.warmup(x, BLOCK=block, grid=(1,))
    steps["warmup"] = {"ok": True, "type": type(compiled).__name__}
except Exception:
    steps["warmup"] = {"ok": False, "traceback": traceback.format_exc()[-1200:]}

if compiled is not None:
    launcher = compiled.run
    steps["launcher_type"] = {"ok": True, "value": type(launcher).__name__}
    steps["getattr_compile_only"] = {"ok": True, "value": repr(getattr(launcher, "compile_only", "MISSING"))}
    steps["getattr_msprof"] = {"ok": True, "value": repr(getattr(launcher, "enable_msprof_register_tensor", "MISSING"))}
    steps["getattr_debug_enabled"] = {"ok": True, "value": repr(getattr(launcher.metadata, "debug_enabled", "MISSING"))}
    rec("call_attr", lambda: launcher.launch)
    rec("function_attr", lambda: compiled.function)
    rec("packed_metadata_attr", lambda: compiled.packed_metadata)
    rec("resolve_stream", lambda: cand._resolve_stream(x.device.index))
    rec("torch_npu_raw_stream", lambda: torch_npu._C._npu_getCurrentRawStreamNoWait)
    rec("npu_current_stream", lambda: torch.npu.current_stream())
out["fallback_steps"] = steps
save()

# --------------------------------------------------------- component cost table
def micro(fn, iters=200000, warm=1000):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e6


numel_m = torch.Tensor.numel
dptr_m = torch.Tensor.data_ptr
contig_m = torch.Tensor.is_contiguous
_FAST = cand._FAST
FAST_GET = _FAST.get
stream = cand._resolve_stream(x.device.index)
addr_type = cand._hgkda_address_type(n)
key = (n, ptr & 15)
call = compiled.run.launch if compiled is not None else None
function = compiled.function if compiled is not None else None
metadata = compiled.packed_metadata if compiled is not None else None

comp = {}
comp["x.numel()"] = micro(lambda: x.numel(), 100000)
comp["Tensor.numel(x)"] = micro(lambda: numel_m(x), 100000)
comp["x.data_ptr()"] = micro(lambda: x.data_ptr(), 100000)
comp["Tensor.data_ptr(x)"] = micro(lambda: dptr_m(x), 100000)
comp["x.is_contiguous()"] = micro(lambda: x.is_contiguous(), 100000)
comp["Tensor.is_contiguous(x)"] = micro(lambda: contig_m(x), 100000)
comp["x.dtype"] = micro(lambda: x.dtype, 100000)
comp["dict.get((n,p))"] = micro(lambda: FAST_GET((n, ptr & 15)), 200000)
comp["addr_type(ptr)"] = micro(lambda: addr_type(ptr), 100000)
comp["stream()"] = micro(lambda: stream(), 100000)
comp["x.numel()+data_ptr()+is_contig()"] = micro(
    lambda: (numel_m(x), dptr_m(x), contig_m(x)), 100000
)
out["component_us"] = comp
save()

# --------------------------------------------------------- assembled variants
def timeit_host(fn, iters=400, warm=50):
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


var = {}
var["candidate_run"] = timeit_host(lambda: cand.run(x))
var["torch_selu"] = timeit_host(lambda: torch.selu_(x))

if call is not None:
    s = stream()

    def launch_floor(p, grid=1, call=call, s=s, f=function, m=metadata):
        call(grid, 1, 1, s, f, m, None, None, None, p, None)

    var["floor_only"] = timeit_host(lambda: launch_floor(ptr))

    def launch_floor_addr(p, grid=1, call=call, s=s, f=function, m=metadata, at=addr_type):
        call(grid, 1, 1, s, f, m, None, None, None, at(p), None)

    var["floor_addr"] = timeit_host(lambda: launch_floor_addr(ptr))

    def minimal(A, grid=1, call=call, s=s, f=function, m=metadata, at=addr_type, get=FAST_GET, dtype=x.dtype):
        p = dptr_m(A)
        if A.numel() == 1024 and dtype is A.dtype:
            call(grid, 1, 1, s, f, m, None, None, None, at(p), None)
            return A
        return cand._dispatch(A, A.numel(), p)

    var["minimal_inline"] = timeit_host(lambda: minimal(x))
out["assembled_us"] = var
save()

print(json.dumps(out, indent=2, default=str), flush=True)
