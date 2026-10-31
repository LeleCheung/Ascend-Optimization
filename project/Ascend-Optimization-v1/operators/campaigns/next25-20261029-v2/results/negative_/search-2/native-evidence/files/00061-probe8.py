"""Controlled A/B in one process: round-3 candidate (tensor-arg, driver stream
lookup) versus the round-4 candidate (int pointer, torch stream, pre-bound
closure).  Measurements are interleaved to cancel slow drift.  Diagnostics only.
"""

import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

WORK = Path(__file__).resolve().parents[2]
ART = Path(os.environ.get("KGS_DEBUG_ARTIFACTS", ".")) / "probe8.json"
DEV = "npu:0"
out = {"ab": {}, "notes": []}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bench(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(3):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) / iters * 1e3)
    ts.sort()
    return ts[1]


def main():
    old = load("old_main", WORK / ".kernelgen/evals/round-0003/main.py")
    new = load("new_main", WORK / "tmp/main.py")

    # correctness spot-check of both modules on identical inputs
    for name, mod in (("old", old), ("new", new)):
        ok = True
        for dt in (torch.float16, torch.bfloat16, torch.float32):
            for shape in ([1], [63], [1024], [1025], [5000], [1048576], [4096, 4096]):
                x = torch.randn(shape, dtype=dt, device=DEV)
                ref = x.clone().neg_()
                mod.run(x)
                torch.npu.synchronize()
                if not torch.equal(x, ref):
                    ok = False
                    out["notes"].append(f"{name} mismatch {dt} {shape}")
                del x, ref
        out["ab"][name + "_correct"] = ok

    shapes = [("t0_fp16_1k", "float16", [1024], 300),
              ("t2_fp16_1M", "float16", [1048576], 200),
              ("t3_fp16_16M", "float16", [4096, 4096], 60),
              ("t5_bf16_64k", "bfloat16", [65536], 200),
              ("t11_fp32_16M", "float32", [4096, 4096], 60)]
    for tag, dt, shape, iters in shapes:
        a = torch.zeros(shape, dtype=getattr(torch, dt), device=DEV)
        rec = {}
        for rnd in range(2):                      # interleave old/new twice
            rec.setdefault("old", []).append(round(bench(lambda: old.run(a), iters), 5))
            rec.setdefault("new", []).append(round(bench(lambda: new.run(a), iters), 5))
        rec["old"] = min(rec["old"])
        rec["new"] = min(rec["new"])
        rec["gain"] = round(rec["old"] / rec["new"], 3) if rec["new"] else None
        out["ab"][tag] = rec
        del a


try:
    main()
except Exception:
    out["fatal"] = traceback.format_exc()
finally:
    ART.parent.mkdir(parents=True, exist_ok=True)
    ART.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str)[:5000])
