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


def safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as exc:  # noqa: BLE001
        return "ERR:" + repr(exc)[:400]


# ---- warm up once so caches are populated
dt = torch.float16
x = torch.randn((4, 32, 32, 32), device="npu", dtype=dt)
c = 32
w = torch.randn([c], device="npu", dtype=dt)
b = torch.randn([c], device="npu", dtype=dt)
args = (x, w, b, 4, 32, 1024, 8, 1e-5)
res = cand.run(*args)
torch.npu.synchronize()
out["cand_cache_keys"] = len(cand._CK_CACHE)
out["cand_ck_entry_is_false"] = [v is False for v in cand._CK_CACHE.values()]
out["stream_fn"] = repr(cand._STREAM_FN)

kern = cand._gn_kernel
out["jit_type"] = repr(type(kern))
dc = getattr(kern, "device_caches", None)
out["device_caches_type"] = repr(type(dc))
try:
    out["device_keys"] = [str(k) for k in dc.keys()]
    dev = list(dc.keys())[0]
    entry = dc[dev]
    out["entry_type"] = repr(type(entry))
    out["entry_len"] = len(entry) if hasattr(entry, "__len__") else "n/a"
    cache = entry[0] if isinstance(entry, (tuple, list)) else entry
    out["cache_type"] = repr(type(cache))
    out["cache_len"] = len(cache)
    key0 = list(cache.keys())[0]
    ck = cache[key0]
    out["ck_type"] = repr(type(ck))
    out["ck_attrs"] = [a for a in dir(ck) if not a.startswith("_")][:80]
    out["has_launch_attr"] = hasattr(ck, "launch")
    out["run_repr"] = repr(ck.run)[:300]
    out["run_sig"] = safe(lambda: str(inspect.signature(ck.run)))
    out["getitem_src"] = safe(lambda: inspect.getsource(type(ck).__getitem__))[:2000]
    out["run_src"] = safe(lambda: inspect.getsource(type(ck).run))[:2000] if inspect.isfunction(ck.run) else "not-py"
    out["launch_src"] = safe(lambda: inspect.getsource(type(ck).launch))[:2000] if hasattr(ck, "launch") else ""
    out["function"] = repr(ck.function)
    out["packed_metadata"] = repr(ck.packed_metadata)[:300]
    out["num_warps"] = safe(lambda: ck.metadata.num_warps)
    out["shared"] = safe(lambda: ck.metadata.shared)
    out["name"] = safe(lambda: ck.name)
    out["n_regs"] = safe(lambda: ck.n_regs)
    global CK
    CK = ck
    global KEY0
    KEY0 = key0
except Exception as exc:  # noqa: BLE001
    out["device_caches_error"] = repr(exc)[:500]

# ---- timings of individual host operations
def timeit(fn, iters=200, warm=20):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


rec = {}
rec["full_cand_run_us"] = timeit(lambda: cand.run(*args))
rec["ref_us"] = timeit(lambda: torch.native_group_norm(x, w, b, 4, 32, 1024, 8, 1e-5))
rec["empty_2_us"] = timeit(lambda: (torch.empty(x.shape, dtype=dt, device=x.device),
                                   torch.empty((2, 4, 8), dtype=dt, device=x.device)))
o = torch.empty(x.shape, dtype=dt, device=x.device)
buf = torch.empty((2, 4, 8), dtype=dt, device=x.device)
m0 = buf[0]
r0 = buf[1]
rec["view2_us"] = timeit(lambda: (buf[0], buf[1]))
rec["dataptr6_us"] = timeit(lambda: (x.data_ptr(), o.data_ptr(), m0.data_ptr(), r0.data_ptr(), w.data_ptr(), b.data_ptr()))

hw, cg, group, ns, cs, ss, count, block = 1024, 4, 8, 32768, 1024, 1, 4096, 1024
launch_args = (x, o, m0, r0, w, b, 1e-5, hw, cg, group, ns, cs, ss, count, block, 1, 1)
grid = (group, 4, 1)


def plain():
    cand._gn_kernel[grid[:2]](*launch_args[:-2], HW=hw, CG=cg, G=group, NS=ns, CS=cs, SS=ss,
                              COUNT=count, BLOCK=block, HAS_AFFINE=True, EVEN=True)


rec["plain_jit_launch_us"] = timeit(plain)

try:
    ck = CK
    stream = cand._stream_ptr()

    def via_getitem():
        ck[(group, 4, 1)](*launch_args)

    rec["ck_getitem_us"] = timeit(via_getitem)

    def via_run():
        ck.run(group, 4, 1, stream, ck.function, ck.packed_metadata, None, None, None, *launch_args)

    rec["ck_run_us"] = timeit(via_run)
    o.zero_()
    torch.npu.synchronize()
    via_run()
    torch.npu.synchronize()
    ref = torch.native_group_norm(x, w, b, 4, 32, 1024, 8, 1e-5)
    rec["via_run_maxdiff_out"] = (o.float() - ref[0].float()).abs().max().item()
    rec["via_run_maxdiff_mean"] = (m0.float() - ref[1].float()).abs().max().item()

    def via_launch():
        ck.launch(group, 4, 1, stream, *launch_args)

    if hasattr(ck, "launch"):
        try:
            rec["ck_launch_us"] = timeit(via_launch)
        except Exception as exc:  # noqa: BLE001
            rec["ck_launch_err"] = repr(exc)[:300]
except Exception as exc:  # noqa: BLE001
    rec["ck_error"] = repr(exc)[:500]

out["timings"] = rec

# grid / device props
d = torch.npu.current_device()
out["device_idx"] = d
out["device_name"] = torch.npu.get_device_name(d)
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
print(json.dumps(out, indent=1)[:12000])
