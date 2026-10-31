"""Bounded diagnostic: load-robust A/B of per-call cost components (min-of-reps)."""
import json
import os
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import importlib.util

spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)

out = {}
rec = {}


def bench(name, fn, reps=120, rounds=3, sync_each=False):
    best = None
    for _ in range(rounds):
        for _ in range(10):
            fn()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        if sync_each:
            for _ in range(reps):
                fn()
                torch.npu.synchronize()
        else:
            for _ in range(reps):
                fn()
            torch.npu.synchronize()
        dt = (time.perf_counter() - t0) / reps * 1e6
        best = dt if best is None else min(best, dt)
    rec[name] = best
    return best


@triton.jit
def _noop1(A):
    pass


@triton.jit
def _noop10(A, B, C, D, E, F, G, H, I, J):
    pass


def get_ck(jitfn, args):
    jitfn[(1,)](*args)
    dc = jitfn.device_caches
    cache = dc[list(dc.keys())[0]][0]
    return list(cache.values())[0]


p = torch.zeros(16, device="npu")
ck1 = get_ck(_noop1, (p,))
ck1.run(1, 1, 1, cand._stream_ptr(), ck1.function, ck1.packed_metadata, None, None, None, p)
ck10 = get_ck(_noop10, tuple(p for _ in range(10)))
ck10.run(1, 1, 1, cand._stream_ptr(), ck10.function, ck10.packed_metadata, None, None, None,
         *tuple(p for _ in range(10)))

# main shape: timing-0 geometry
x = torch.randn((4, 32, 32, 32), device="npu", dtype=torch.float16)
c = 32
w = torch.randn([c], device="npu", dtype=torch.float16)
b = torch.randn([c], device="npu", dtype=torch.float16)
shp = x.shape
buf_shape = (2, 4, 8)

# warm candidate so its cached raw launcher exists
cand.run(x, w, b, 4, 32, 1024, 8, 1e-5)
torch.npu.synchronize()

# pull the cached entry/launcher and the compiled kernel
e = list(cand._CACHE.values())[0]
launcher = e[0]
ck = list(cand._gn_kernel.device_caches[
    list(cand._gn_kernel.device_caches.keys())[0]][0].values())[0]
out["cached_launcher_type"] = repr(type(launcher))
out["ck_run_launch_type"] = repr(type(ck.run.launch))

o = x.new_empty(shp)
mb = x.new_empty(buf_shape)
m0, r0 = mb.unbind(0)
sptr = cand._resolve_stream()


def call_launcher():
    launcher(x, o, m0, r0, w, b)


def noop1_launch():
    ck1.run(1, 1, 1, sptr(), ck1.function, ck1.packed_metadata, None, None, None, p)


def noop10_launch():
    ck10.run(1, 1, 1, sptr(), ck10.function, ck10.packed_metadata, None, None, None,
             p, p, p, p, p, p, p, p, p, p)


def allocs_new_empty():
    t1 = x.new_empty(shp)
    t2 = x.new_empty(buf_shape)
    return t2.unbind(0)


def allocs_torch_empty():
    t1 = torch.empty(shp, dtype=torch.float16, device=x.device)
    t2 = torch.empty(buf_shape, dtype=torch.float16, device=x.device)
    return t2[0], t2[1]


def allocs_one_tensor():
    return x.new_empty(shp)


def allocs_unbind_only():
    return mb.unbind(0)


bench("ref", lambda: torch.native_group_norm(x, w, b, 4, 32, 1024, 8, 1e-5))
bench("launch_only", call_launcher)
bench("launch_only_sync", call_launcher, sync_each=True)
bench("noop1_launch", noop1_launch)
bench("noop10_launch", noop10_launch)
bench("alloc_new_empty", allocs_new_empty)
bench("alloc_torch_empty", allocs_torch_empty)
bench("alloc_one_new_empty", allocs_one_tensor)
bench("alloc_unbind_only", allocs_unbind_only)
bench("wrapper_v2", lambda: cand.run(x, w, b, 4, 32, 1024, 8, 1e-5))
bench("wrapper_v2_sync", lambda: cand.run(x, w, b, 4, 32, 1024, 8, 1e-5), sync_each=True)
bench("alloc_plus_launch", lambda: (lambda t1, t2, mm: launcher(x, t1, mm[0], mm[1], w, b))(
    x.new_empty(shp), x.new_empty(buf_shape), mb.unbind(0)))

# bigger geometry (timing-2) for size sensitivity of the launch path
x2 = torch.randn((16, 128, 32, 32), device="npu", dtype=torch.float16)
w2 = torch.randn([128], device="npu", dtype=torch.float16)
b2 = torch.randn([128], device="npu", dtype=torch.float16)
cand.run(x2, w2, b2, 16, 128, 1024, 8, 1e-5)
torch.npu.synchronize()
o2 = x2.new_empty(x2.shape)
mb2 = x2.new_empty((2, 16, 8))
m2, r2 = mb2.unbind(0)
lk2 = list(cand._CACHE.values())[1][0]
bench("ref_big", lambda: torch.native_group_norm(x2, w2, b2, 16, 128, 1024, 8, 1e-5), reps=60)
bench("wrapper_big", lambda: cand.run(x2, w2, b2, 16, 128, 1024, 8, 1e-5), reps=60)
bench("launch_only_big", lambda: lk2(x2, o2, m2, r2, w2, b2), reps=60)

out["timings"] = rec
art = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
os.makedirs(art, exist_ok=True)
with open(os.path.join(art, "ab.json"), "w") as f:
    json.dump(out, f, indent=1)
print("RESULT", json.dumps(out, indent=1))
