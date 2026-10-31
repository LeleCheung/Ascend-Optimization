"""Bounded target diagnostic: what does the cached low-level launch cost consist of?"""
import importlib.util
import json
import os
import time

import torch
import torch_npu
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
RES = {}


def load_candidate():
    spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cand = load_candidate()
DEV = torch.npu.current_device()


@triton.jit
def _noop(P):
    tl.store(P + tl.program_id(0), 0)


g = torch.Generator().manual_seed(1729)
A = torch.randn(1024, generator=g).to(device=DEV, dtype=torch.float16)
B = (torch.randn(1024, generator=g).abs() + 0.05).to(device=DEV, dtype=torch.float16)
cand.run(A, B)
torch.npu.synchronize()

key = (A.dtype, B.dtype, A.device.index, 1024)
launcher, fnt, pmd = cand._ENTRIES[key]
RES["launcher_type"] = f"{type(launcher)}"
RES["launcher_mro"] = [t.__name__ for t in type(launcher).__mro__]
RES["launch_attr_type"] = f"{type(getattr(launcher, 'launch', None))}"
RES["launcher_dir"] = [d for d in dir(launcher) if not d.startswith('__')]
RES["run_obj_type"] = f"{type(getattr(cand._fmod_flat, 'warmup', None))}"
try:
    import inspect
    RES["launch_src"] = inspect.getsource(type(launcher))
except Exception as exc:
    RES["launch_src"] = f"unavailable: {type(exc).__name__}: {str(exc)[:200]}"
try:
    md = launcher.metadata
    RES["metadata_fields"] = {k: str(getattr(md, k))[:120] for k in dir(md)
                              if not k.startswith('_') and not callable(getattr(md, k))}
except Exception as exc:
    RES["metadata_fields"] = f"unavailable: {type(exc).__name__}"


def time_loop(fn, iters=400):
    for _ in range(20):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def time_event(fn, iters=300):
    for _ in range(20):
        fn()
    torch.npu.synchronize()
    s = torch.npu.Event(enable_timing=True)
    e = torch.npu.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.npu.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


pa, pb = A.data_ptr(), B.data_ptr()
raw = torch_npu._C._npu_getCurrentRawStreamNoWait
stream = raw(DEV)
grid = 1

# current form
RES["cur_launch"] = (time_loop(lambda: launcher.launch(grid, 1, 1, raw(DEV), fnt, pmd,
                                                       None, None, None, pa, pb, 1024, 4096)),
                     time_event(lambda: launcher.launch(grid, 1, 1, raw(DEV), fnt, pmd,
                                                        None, None, None, pa, pb, 1024, 4096)))
# without python-side stream refresh
RES["launch_fixed_stream"] = (time_loop(lambda: launcher.launch(grid, 1, 1, stream, fnt, pmd,
                                                               None, None, None, pa, pb, 1024, 4096)),)
# callable launcher (standard triton __call__ signature)
if callable(launcher):
    try:
        RES["callable_launch"] = (time_loop(lambda: launcher(grid, 1, 1, stream, fnt, pmd,
                                                             None, None, None, pa, pb, 1024, 4096)),)
    except Exception as exc:
        RES["callable_launch"] = f"{type(exc).__name__}: {str(exc)[:200]}"
# fewer args (drop constexpr scalars)
for name, args in (("no_constexpr_args", (pa, pb)),
                   ("one_tensor_arg", (pa,))):
    try:
        launcher.launch(grid, 1, 1, stream, fnt, pmd, None, None, None, *args)
        torch.npu.synchronize()
        RES[f"launch_{name}"] = (time_loop(lambda: launcher.launch(grid, 1, 1, stream, fnt, pmd,
                                                                    None, None, None, *args)),)
    except Exception as exc:
        RES[f"launch_{name}"] = f"{type(exc).__name__}: {str(exc)[:200]}"
# stream None
try:
    launcher.launch(grid, 1, 1, 0, fnt, pmd, None, None, None, pa, pb, 1024, 4096)
    torch.npu.synchronize()
    RES["launch_stream_zero"] = (time_loop(lambda: launcher.launch(grid, 1, 1, 0, fnt, pmd,
                                                                   None, None, None, pa, pb, 1024, 4096)),)
except Exception as exc:
    RES["launch_stream_zero"] = f"{type(exc).__name__}: {str(exc)[:200]}"

# empty kernel through the same cached direct-launch machinery (host floor)
buf = torch.empty(64, dtype=torch.float32, device=DEV)
ncomp = _noop.warmup(buf, 1, grid=(1,), enable_fp_fusion=False)
nl = ncomp.run
RES["noop_direct"] = (time_loop(lambda: nl.launch(1, 1, 1, raw(DEV), ncomp.function,
                                                  ncomp.packed_metadata, None, None, None,
                                                  buf.data_ptr())),)
# host cost of the pieces
RES["stream_getter_ns"] = (lambda: (lambda t0: (time.perf_counter() - t0) / 20000 * 1e9)(
    (lambda: ([(torch_npu._C._npu_getCurrentRawStreamNoWait(DEV)) for _ in range(1)],
              time.perf_counter())[1])()))()
t0 = time.perf_counter()
for _ in range(20000):
    torch_npu._C._npu_getCurrentRawStreamNoWait(DEV)
RES["stream_getter_ns"] = (time.perf_counter() - t0) / 20000 * 1e9
t0 = time.perf_counter()
for _ in range(20000):
    A.get_device()
RES["get_device_ns"] = (time.perf_counter() - t0) / 20000 * 1e9
t0 = time.perf_counter()
for _ in range(20000):
    A.shape == B.shape
RES["shape_eq_ns"] = (time.perf_counter() - t0) / 20000 * 1e9
t0 = time.perf_counter()
for _ in range(20000):
    _ = (A.dtype, B.dtype, A.device.index, 1024)
RES["tuple_key_ns"] = (time.perf_counter() - t0) / 20000 * 1e9
t0 = time.perf_counter()
for _ in range(20000):
    A.fmod_(B)
torch.npu.synchronize()
RES["ref_per_call_us"] = (time.perf_counter() - t0) / 20000 * 1e6

with open(os.path.join(ART, "probe2.json"), "w") as fh:
    json.dump(RES, fh, indent=1)
print(json.dumps(RES, indent=1)[:6000])
