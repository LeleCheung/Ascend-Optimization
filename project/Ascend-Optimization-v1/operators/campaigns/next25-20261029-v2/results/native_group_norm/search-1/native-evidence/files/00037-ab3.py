"""Bounded diagnostic: which compiled entry wins, and what each per-call component costs."""
import json
import os
import time

import torch
import torch_npu  # noqa: F401
import importlib.util

spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)

out = {}
rec = {}
CAP = {}


def bench(name, fn, reps=100, rounds=3, sync_each=False):
    best = None
    for _ in range(rounds):
        for _ in range(5):
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


# spy on the tuner so we learn exactly what the candidate selected
orig = cand._tune_launch


def spy(ck, grid, args, sget, affine):
    CAP["ck"] = ck
    CAP["grid"] = grid
    CAP["args"] = args
    CAP["sget"] = sget
    res = orig(ck, grid, args, sget, affine)
    CAP["chosen"] = res
    return res


cand._tune_launch = spy

x = torch.randn((4, 32, 32, 32), device="npu", dtype=torch.float16)
w = torch.randn([32], device="npu", dtype=torch.float16)
b = torch.randn([32], device="npu", dtype=torch.float16)
o, m, r = cand.run(x, w, b, 4, 32, 1024, 8, 1e-5)
torch.npu.synchronize()
cand._tune_launch = orig

out["captured"] = bool(CAP)
ck = CAP["ck"]
gx, gy, gz = CAP["grid"]
args = CAP["args"]
sget = CAP["sget"]
stream_int = sget()
out["grid"] = [gx, gy, gz]
out["n_args"] = len(args)
out["ck_type"] = repr(type(ck))
out["run_type"] = repr(type(ck.run))
out["launch_type"] = repr(type(ck.run.launch))
fnp = ck.function
pm = ck.packed_metadata

entry_run = ck.run
entry_launch = ck.run.launch


def call_run():
    entry_run(gx, gy, gz, sget(), fnp, pm, None, None, None, *args)


def call_launch():
    entry_launch(gx, gy, gz, sget(), fnp, pm, None, None, None, *args)


def call_launch_cached_stream():
    entry_launch(gx, gy, gz, stream_int, fnp, pm, None, None, None, *args)


bench("entry_run_batch", call_run)
bench("entry_launch_batch", call_launch)
bench("entry_run_sync", call_run, sync_each=True)
bench("entry_launch_sync", call_launch, sync_each=True)
bench("entry_launch_cachedstream_batch", call_launch_cached_stream)
bench("stream_getter_only", lambda: sget())
bench("wrapper_batch", lambda: cand.run(x, w, b, 4, 32, 1024, 8, 1e-5))
bench("wrapper_sync", lambda: cand.run(x, w, b, 4, 32, 1024, 8, 1e-5), sync_each=True)
bench("ref_batch", lambda: torch.native_group_norm(x, w, b, 4, 32, 1024, 8, 1e-5))
bench("ref_sync", lambda: torch.native_group_norm(x, w, b, 4, 32, 1024, 8, 1e-5),
      sync_each=True)
bench("none_only", lambda: None)

out["timings"] = rec
art = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
os.makedirs(art, exist_ok=True)
with open(os.path.join(art, "ab3.json"), "w") as f:
    json.dump(out, f, indent=1)
print("RESULT", json.dumps(out, indent=1))
