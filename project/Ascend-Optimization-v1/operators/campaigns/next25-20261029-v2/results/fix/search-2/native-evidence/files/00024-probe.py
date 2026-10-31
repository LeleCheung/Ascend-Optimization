"""Bounded diagnostic: host launch path + per-call cost breakdown vs torch.fix."""
import json
import os
import time
import importlib.util

import torch
import torch_npu

ARTS = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
res = {"torch": torch.__version__}

spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)

# ---- target facts -------------------------------------------------------
facts = {}
try:
    from driver.active import utils as drv_utils
    dev = torch.npu.current_device()
    try:
        facts["device_props"] = drv_utils.get_device_properties(dev)
    except Exception as e:  # noqa
        facts["device_props_err"] = repr(e)
    facts["driver_utils_attrs"] = [a for a in dir(drv_utils) if not a.startswith("_")]
except Exception as e:  # noqa
    facts["driver_import_err"] = repr(e)
facts["npu_device_count"] = torch.npu.device_count()
facts["current_device"] = torch.npu.current_device()
res["facts"] = facts


def host_loop(fn, iters, warm=5):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    t2 = time.perf_counter()
    return {"submit_us": (t1 - t0) / iters * 1e6,
            "wall_us": (t2 - t0) / iters * 1e6}


shapes = {"1024": 1024, "1M": 1048576, "16M": 4096 * 4096}
dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

res["perf"] = {}
for dn, dt in dtypes.items():
    for sn, n in shapes.items():
        key = f"{dn}_{sn}"
        x = torch.randn(n, device="npu:0", dtype=dt)
        entry = {}


        def call_cand():
            return cand.run(x)


        def call_ref():
            return torch.fix(x)


        entry["cand"] = host_loop(call_cand, 50 if n > 2e6 else 200)
        entry["ref"] = host_loop(call_ref, 50 if n > 2e6 else 200)
        # check fast launch path availability
        try:
            out = torch.empty_like(x)
            compiled = cand._fix_contiguous.warmup(x, out, n, BLOCK=4096, grid=(1,))
            launcher = compiled.run
            entry["launch_attr"] = hasattr(launcher, "launch")
            entry["compile_only"] = bool(getattr(launcher, "compile_only", None))
            entry["msprof_flag"] = bool(getattr(launcher, "enable_msprof_register_tensor", None))
            entry["debug_flag"] = bool(getattr(launcher.metadata, "debug_enabled", None))
            entry["debug_flag2"] = bool(getattr(launcher.metadata, "debug", None))
            entry["pkg_metadata_type"] = type(compiled.packed_metadata).__name__
        except Exception as e:  # noqa
            entry["warmup_err"] = repr(e)[:300]
        entry["cached_keys"] = [str(k) for k in list(cand._ENTRIES.keys())[:12]]
        entry["cached_launch"] = [str(v[2] is not None) for v in list(cand._ENTRIES.values())[:12]]
        # correctness spot check incl. -0 and specials
        xs = torch.tensor([0.0, -0.0, float("nan"), float("inf"), -float("inf"),
                           -1.0, 1.0, -0.5, 0.5, -1.5, 2.7, -2.7, 1024.5, -0.0001],
                          dtype=dt, device="npu:0")
        got = cand.run(xs)
        exp = torch.fix(xs)
        entry["spot_equal"] = bool(torch.equal(got.view(torch.int16 if dt != torch.float32 else torch.int32),
                                              exp.view(torch.int16 if dt != torch.float32 else torch.int32)))
        entry["spot_got"] = [float(v) for v in got.tolist()]
        entry["spot_exp"] = [float(v) for v in exp.tolist()]
        res["perf"][key] = entry

# cost of an empty allocation alone
x = torch.randn(1024, device="npu:0", dtype=torch.float16)
res["alloc"] = {
    "empty_like_mf": host_loop(lambda: torch.empty_like(x, memory_format=torch.contiguous_format), 200),
    "empty_like": host_loop(lambda: torch.empty_like(x), 200),
}
f = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
res["stream"] = {"has_raw": f is not None,
                 "raw": host_loop(lambda: f(0), 200) if f else None}

os.makedirs(ARTS, exist_ok=True)
with open(os.path.join(ARTS, "probe.json"), "w") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res)[:4000])
