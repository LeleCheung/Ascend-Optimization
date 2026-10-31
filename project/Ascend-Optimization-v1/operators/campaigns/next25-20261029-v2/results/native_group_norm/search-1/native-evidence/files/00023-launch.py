"""Bounded diagnostic: find the cheapest host launch path for the compiled kernel."""
import inspect
import json
import os
import sys
import time

import torch
import torch_npu  # noqa: F401
import importlib.util

spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)

out = {}
srcs = {}


def safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as exc:  # noqa: BLE001
        return "ERR:" + repr(exc)[:300]


dt = torch.float16
x = torch.randn((4, 32, 32, 32), device="npu", dtype=dt)
c = 32
w = torch.randn([c], device="npu", dtype=dt)
b = torch.randn([c], device="npu", dtype=dt)
res = cand.run(x, w, b, 4, 32, 1024, 8, 1e-5)
torch.npu.synchronize()
out["cand_cache_len"] = len(cand._CK_CACHE)
out["cand_ck_entry_is_false"] = [v is False for v in cand._CK_CACHE.values()]
out["stream_fn"] = repr(cand._STREAM_FN)

kern = cand._gn_kernel
out["jit_type"] = repr(type(kern))
dc = getattr(kern, "device_caches", None)
CK = None
try:
    dev = list(dc.keys())[0]
    entry = dc[dev]
    out["entry_type"] = repr(type(entry))
    out["entry_len"] = len(entry) if hasattr(entry, "__len__") else "n/a"
    cache = entry[0] if isinstance(entry, (tuple, list)) else entry
    out["cache_type"] = repr(type(cache))
    out["cache_len"] = len(cache)
    key0 = list(cache.keys())[0]
    out["cache_key0"] = repr(key0)[:400]
    ck = cache[key0]
    CK = ck
    out["ck_type"] = repr(type(ck))
    out["ck_attrs"] = [a for a in dir(ck) if not a.startswith("__")][:100]
    out["run_repr"] = repr(ck.run)[:300]
    out["run_sig"] = safe(lambda: str(inspect.signature(ck.run)))
    out["getitem_src"] = safe(lambda: inspect.getsource(type(ck).__getitem__))[:2500]
    out["run_src"] = safe(lambda: inspect.getsource(ck.run))[:3000]
    out["function"] = repr(ck.function)
    out["packed_metadata"] = repr(ck.packed_metadata)[:400]
    out["metadata_repr"] = repr(ck.metadata)[:600]
    out["name"] = safe(lambda: ck.name)
    import triton.spec.ascend.runtime.jit as ajit

    jsrc = inspect.getsource(ajit)
    srcs["ajit_len"] = len(jsrc)
    lines = jsrc.splitlines()
    # dump the run() body region
    idx = [i for i, ln in enumerate(lines) if ln.strip().startswith("def run(")]
    out["ajit_run_def_lines"] = idx
    if idx:
        i0 = idx[0]
        srcs["ajit_run_body"] = "\n".join(lines[i0:i0 + 140])
except Exception as exc:  # noqa: BLE001
    out["device_caches_error"] = repr(exc)[:500]


def timeit(fn, iters=200, warm=20):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


o = torch.empty(x.shape, dtype=dt, device=x.device)
buf = torch.empty((2, 4, 8), dtype=dt, device=x.device)
m0 = buf[0]
r0 = buf[1]
ref = torch.native_group_norm(x, w, b, 4, 32, 1024, 8, 1e-5)

rec = {}
rec["full_cand_run_us"] = timeit(lambda: cand.run(x, w, b, 4, 32, 1024, 8, 1e-5))
rec["ref_us"] = timeit(lambda: torch.native_group_norm(x, w, b, 4, 32, 1024, 8, 1e-5))
rec["empty_2_us"] = timeit(lambda: (torch.empty(x.shape, dtype=dt, device=x.device),
                                   torch.empty((2, 4, 8), dtype=dt, device=x.device)))
rec["view2_us"] = timeit(lambda: (buf[0], buf[1]))
rec["dataptr6_us"] = timeit(lambda: (x.data_ptr(), o.data_ptr(), m0.data_ptr(), r0.data_ptr(),
                                    w.data_ptr(), b.data_ptr()))

hw, cg, group, ns, cs, ss, count, block = 1024, 4, 8, 32768, 1024, 1, 4096, 1024
ptrs = (x.data_ptr(), o.data_ptr(), m0.data_ptr(), r0.data_ptr(), w.data_ptr(), b.data_ptr())
ten = (x, o, m0, r0, w, b)
consts = (hw, cg, group, ns, cs, ss, count, block, 1, 1)


def check(tag):
    torch.npu.synchronize()
    d = (o.float() - ref[0].float()).abs().max().item()
    dm = (m0.float() - ref[1].float()).abs().max().item()
    rec[tag + "_maxdiff"] = [d, dm]


def plain():
    cand._gn_kernel[(group, 4)](x, o, m0, r0, w, b, 1e-5, HW=hw, CG=cg, G=group, NS=ns, CS=cs,
                                SS=ss, COUNT=count, BLOCK=block, HAS_AFFINE=True, EVEN=True)


o.zero_()
rec["plain_jit_launch_us"] = timeit(plain)
check("plain")

if CK is not None:
    ck = CK
    stream = cand._stream_ptr()
    fn_p = ck.function
    pm = ck.packed_metadata

    def runA():  # all args (incl. constexpr) as raw ints, like the seed's cached path
        ck.run(group, 4, 1, stream, fn_p, pm, None, None, None, *ptrs, 1e-5, *consts)

    o.zero_()
    try:
        rec["ck_run_allargs_us"] = timeit(runA)
        check("runA")
    except Exception as exc:  # noqa: BLE001
        rec["ck_run_allargs_err"] = repr(exc)[:300]

    def runB():  # only non-constexpr args
        ck.run(group, 4, 1, stream, fn_p, pm, None, None, None, *ptrs, 1e-5)

    o.zero_()
    try:
        rec["ck_run_nonconst_us"] = timeit(runB)
        check("runB")
    except Exception as exc:  # noqa: BLE001
        rec["ck_run_nonconst_err"] = repr(exc)[:300]

    def runC():  # tensors instead of ints
        ck.run(group, 4, 1, stream, fn_p, pm, None, None, None, *ten, 1e-5, *consts)

    o.zero_()
    try:
        rec["ck_run_tensors_us"] = timeit(runC)
        check("runC")
    except Exception as exc:  # noqa: BLE001
        rec["ck_run_tensors_err"] = repr(exc)[:300]

    o.zero_()
    try:
        def gi():
            ck[(group, 4, 1)](*ptrs, 1e-5, *consts)
        rec["ck_getitem_us"] = timeit(gi)
        check("getitem")
    except Exception as exc:  # noqa: BLE001
        rec["ck_getitem_err"] = repr(exc)[:300]

    if hasattr(ck, "launch"):
        try:
            def lz():
                ck.launch(group, 4, 1, stream, *ptrs, 1e-5, *consts)
            o.zero_()
            rec["ck_launch_us"] = timeit(lz)
            check("launch")
        except Exception as exc:  # noqa: BLE001
            rec["ck_launch_err"] = repr(exc)[:300]

out["timings"] = rec
out["srcs"] = srcs

d = torch.npu.current_device()
out["device_name"] = safe(lambda: torch.npu.get_device_name(d))
try:
    from driver.active import utils as drv_utils

    props = drv_utils.get_device_properties(d)
    out["props"] = {k: str(v) for k, v in dict(props).items()} if not isinstance(props, str) else props
except Exception as exc:  # noqa: BLE001
    out["props_err"] = repr(exc)[:300]

art = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
os.makedirs(art, exist_ok=True)
with open(os.path.join(art, "launch_probe.json"), "w") as f:
    json.dump(out, f, indent=1)
print("TIMINGS", json.dumps(rec, indent=1))
print("META", json.dumps({k: v for k, v in out.items() if k not in ("timings", "srcs")}, indent=1)[:4000])
