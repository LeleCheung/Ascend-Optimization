"""Diagnostic: is the candidate's fast launch path engaging, and what is its floor?"""
import importlib.util
import json
import os
import statistics
import sys
import time

import torch

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = {}

spec = importlib.util.spec_from_file_location("cand", os.path.join(os.getcwd(), "tmp", "main.py"))
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)

OUT["has_compute_cache_key"] = cand._compute_cache_key is not None
OUT["slow_before"] = list(cand._SLOW)


def bench_percall(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return round(statistics.median(ts), 2)


def bench_batch(fn, iters=300, warmup=30):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return round((time.perf_counter() - t0) * 1e6 / iters, 2)


res = {}
for label, shape, dt in [("fp16_1024", (1024,), torch.float16),
                         ("fp16_65536", (65536,), torch.float16),
                         ("fp16_1048576", (1048576,), torch.float16),
                         ("fp32_1024", (1024,), torch.float32),
                         ("fp16_4096x4096", (4096, 4096), torch.float16)]:
    x = torch.randn(shape, device="npu:0", dtype=dt)
    r = {}
    r["slow_before"] = list(cand._SLOW)
    r["cand_percall"] = bench_percall(lambda: cand.run(x))
    r["slow_after"] = list(cand._SLOW)
    r["cand_batch"] = bench_batch(lambda: cand.run(x))
    ref = torch.randn(shape, device="npu:0", dtype=dt)
    r["torch_percall"] = bench_percall(lambda: torch.selu_(ref))
    r["torch_batch"] = bench_batch(lambda: torch.selu_(ref))
    # correctness of the in-place result
    a = torch.randn(shape, device="npu:0", dtype=dt)
    b = a.detach().clone()
    out = cand.run(a)
    torch.npu.synchronize()
    ref_out = torch.selu(b)
    r["max_abs_err"] = (out.float() - ref_out.float()).abs().max().item()
    r["same_storage"] = out.data_ptr() == a.data_ptr()
    res[label] = r

OUT["results"] = res
OUT["slow_final"] = list(cand._SLOW)
os.makedirs(ART, exist_ok=True)
with open(os.path.join(ART, "probe_candidate.json"), "w") as fh:
    json.dump(OUT, fh, indent=2, default=str)
print(json.dumps(OUT, indent=2, default=str)[:4000])
