"""Diagnostic: per-step cost of the candidate's cached launch path."""
import importlib.util
import json
import os
import time

import torch

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = {}

spec = importlib.util.spec_from_file_location("cand", os.path.join(os.getcwd(), "tmp", "main.py"))
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)

from triton.runtime import driver as _driver  # noqa: E402


def t(fn, iters=500, warmup=50):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return round((time.perf_counter() - t0) * 1e6 / iters, 2)


x = torch.randn(1024, device="npu:0", dtype=torch.float16)
n = x.numel()
res = {}

cand.run(x)  # compile + populate caches
torch.npu.synchronize()

device = _driver.active.get_current_device()
kernel = cand.selu_kernel_
kc, kkc, _tg, _be, binder = kernel.device_caches[device]
ba, sp, op = binder(x, n, BLOCK_SIZE=1024)
key = cand._compute_cache_key(kkc, sp, op)
compiled = kc.get(key)
OUT["compiled_found"] = compiled is not None
OUT["compiled_type"] = type(compiled).__name__

res["wrap_run_full"] = t(lambda: cand.run(x))
res["wrap_selu_full"] = t(lambda: cand.selu_(x))
res["fast_launch_call"] = t(lambda: cand._fast_launch(kernel, (x, n), 1024, 1))
res["get_current_device"] = t(lambda: _driver.active.get_current_device())
res["device_caches_lookup"] = t(lambda: kernel.device_caches[device])
res["binder"] = t(lambda: binder(x, n, BLOCK_SIZE=1024))
res["binder_plus_key"] = t(lambda: cand._compute_cache_key(kkc, *binder(x, n, BLOCK_SIZE=1024)[1:]))
res["compute_key_only"] = t(lambda: cand._compute_cache_key(kkc, sp, op))
res["kc_get"] = t(lambda: kc.get(key))
res["hasattr_result"] = t(lambda: hasattr(compiled, "result"))
res["get_function"] = t(lambda: compiled.function)
res["get_packed_meta"] = t(lambda: compiled.packed_metadata)
res["get_current_stream"] = t(lambda: _driver.active.get_current_stream(device))

st = _driver.active.get_current_stream(device)
run = compiled.run
fn = compiled.function
pm = compiled.packed_metadata
vals = tuple(ba.values())
res["raw_run_only"] = t(lambda: run(1, 1, 1, st, fn, pm, None, None, None, *vals))


def full_manual():
    d = _driver.active.get_current_device()
    a, b, _c, _d, bnd = kernel.device_caches[d]
    bb, ss, oo = bnd(x, n, BLOCK_SIZE=1024)
    kk = cand._compute_cache_key(b, ss, oo)
    cp = a.get(kk)
    s = _driver.active.get_current_stream(d)
    cp.run(1, 1, 1, s, cp.function, cp.packed_metadata, None, None, None, *bb.values())


res["manual_full"] = t(full_manual)


def manual_no_binder():
    d = 0
    s = _driver.active.get_current_stream(d)
    run(1, 1, 1, s, fn, pm, None, None, None, *vals)


res["manual_no_binder"] = t(manual_no_binder)


def manual_stream_plus_run():
    s = _driver.active.get_current_stream(0)
    run(1, 1, 1, s, fn, pm, None, None, None, x, n, 1024)


res["manual_stream_plus_run_reals"] = t(manual_stream_plus_run)

OUT["steps_us"] = res
os.makedirs(ART, exist_ok=True)
with open(os.path.join(ART, "probe_steps.json"), "w") as fh:
    json.dump(OUT, fh, indent=2, default=str)
print(json.dumps(OUT["steps_us"], indent=2, default=str))
