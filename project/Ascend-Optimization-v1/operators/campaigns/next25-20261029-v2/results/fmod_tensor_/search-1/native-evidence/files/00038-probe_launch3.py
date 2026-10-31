"""Bounded target diagnostic: launch-arg sensitivity and lean host-path prototypes."""
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
_raw = torch_npu._C._npu_getCurrentRawStreamNoWait


@triton.jit
def _noop(P):
    tl.store(P + tl.program_id(0), 0)


g = torch.Generator().manual_seed(1729)
A = torch.randn(1024, generator=g).to(device=DEV, dtype=torch.float16)
B = (torch.randn(1024, generator=g).abs() + 0.05).to(device=DEV, dtype=torch.float16)
Aref = A.clone()
cand.run(A, B)
torch.npu.synchronize()

key = (A.dtype, B.dtype, A.device.index, 1024)
launch, fnt, pmd = cand._ENTRIES[key]
RES["launch_type"] = f"{type(launch)}"
RES["entry_kind"] = "raw C builtin returned by compiled.run.launch"


def time_loop(fn, iters=500):
    for _ in range(30):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


pa, pb = A.data_ptr(), B.data_ptr()
stream = _raw(DEV)

RES["cur_launch_us"] = time_loop(lambda: launch(1, 1, 1, _raw(DEV), fnt, pmd,
                                               None, None, None, pa, pb, 1024, 4096))
RES["fixed_stream_us"] = time_loop(lambda: launch(1, 1, 1, stream, fnt, pmd,
                                                  None, None, None, pa, pb, 1024, 4096))
RES["stream_zero_us"] = None
try:
    launch(1, 1, 1, 0, fnt, pmd, None, None, None, pa, pb, 1024, 4096)
    torch.npu.synchronize()
    RES["stream_zero_us"] = time_loop(lambda: launch(1, 1, 1, 0, fnt, pmd,
                                                    None, None, None, pa, pb, 1024, 4096))
except Exception as exc:
    RES["stream_zero_us"] = f"{type(exc).__name__}: {str(exc)[:160]}"
for name, extra in (("no_scalars", (pa, pb)), ("one_arg", (pa,))):
    try:
        launch(1, 1, 1, stream, fnt, pmd, None, None, None, *extra)
        torch.npu.synchronize()
        RES[name] = time_loop(lambda: launch(1, 1, 1, stream, fnt, pmd, None, None, None, *extra))
    except Exception as exc:
        RES[name] = f"{type(exc).__name__}: {str(exc)[:160]}"

buf = torch.empty(64, dtype=torch.float32, device=DEV)
ncomp = _noop.warmup(buf, 1, grid=(1,), enable_fp_fusion=False)
nlaunch = ncomp.run.launch
RES["noop_direct_us"] = time_loop(lambda: nlaunch(1, 1, 1, stream, ncomp.function,
                                                 ncomp.packed_metadata, None, None, None,
                                                 buf.data_ptr()))

# lean host-path prototypes
_ENT = {}
ent = (launch, fnt, pmd, 1)
_ENT[(A.dtype, DEV, 1024)] = ent


def lean_full(A, B):
    if A.is_contiguous() and B.is_contiguous() and A.shape == B.shape:
        e = _ENT.get((A.dtype, A.get_device(), A.numel()))
        if e is not None:
            e[0](e[3], 1, 1, _raw(A.get_device()), e[1], e[2], None, None, None,
                 A.data_ptr(), B.data_ptr())
            return A
    return cand.run(A, B)


def lean_noshape(A, B):
    if A.is_contiguous() and B.is_contiguous():
        e = _ENT.get((A.dtype, A.get_device(), A.numel()))
        if e is not None:
            e[0](e[3], 1, 1, _raw(A.get_device()), e[1], e[2], None, None, None,
                 A.data_ptr(), B.data_ptr())
            return A
    return cand.run(A, B)


def lean_min(A, B):
    e = _ENT.get((A.dtype, A.get_device(), A.numel()))
    if e is not None:
        e[0](e[3], 1, 1, _raw(A.get_device()), e[1], e[2], None, None, None,
             A.data_ptr(), B.data_ptr())
        return A
    return cand.run(A, B)


RES["lean_full_us"] = time_loop(lambda: lean_full(A, B))
RES["lean_noshape_us"] = time_loop(lambda: lean_noshape(A, B))
RES["lean_min_us"] = time_loop(lambda: lean_min(A, B))
RES["seed_run_us"] = time_loop(lambda: cand.run(A, B))
RES["ref_fmod_us"] = time_loop(lambda: A.fmod_(B))

# component costs
for name, fn in (("stream_getter", lambda: _raw(DEV)),
                 ("get_device", lambda: A.get_device()),
                 ("shape_eq", lambda: A.shape == B.shape),
                 ("tuple_key", lambda: (A.dtype, A.get_device(), 1024)),
                 ("dict_get", lambda: _ENT.get((A.dtype, DEV, 1024)))):
    t0 = time.perf_counter()
    for _ in range(20000):
        fn()
    RES[name + "_ns"] = (time.perf_counter() - t0) / 20000 * 1e9

with open(os.path.join(ART, "probe3.json"), "w") as fh:
    json.dump(RES, fh, indent=1)
print(json.dumps(RES, indent=1))
