"""Validate and time the current candidate end to end, including the cached
raw-launcher path, on every timing geometry plus the fallback domains.

Checks: output dtype/shape, values against the aten reference with the
workload tolerance (atol = rtol = 0.015 for 16-bit, 1e-4 for float32,
required matched ratio 1.0), and bitwise stability across repeated calls
(call 2 uses the cached launcher, call 6 is another).
"""

import json
import os
import statistics
import time

import torch
import torch_npu  # noqa: F401

import sys
sys.path.insert(0, "tmp")
import main as cand  # noqa: E402

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = os.path.join(ART, "validate_candidate.json")
res = {"rows": [], "errors": []}


def flush():
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)


def check(name, x, size, ac=False, sh=None, sw=None):
    tol = 0.015 if x.dtype in (torch.float16, torch.bfloat16) else 1e-4
    ref = torch.ops.aten.upsample_bilinear2d(x, list(size), ac, sh, sw)
    got = cand.run(x, list(size), ac, sh, sw)
    row = {"case": name, "dtype": str(x.dtype), "shape": list(x.shape),
           "out": list(size), "ac": ac, "sh": sh, "sw": sw}
    row["dtype_ok"] = bool(got.dtype == ref.dtype)
    row["shape_ok"] = bool(tuple(got.shape) == tuple(ref.shape))
    d = (got.float() - ref.float()).abs()
    row["max_abs_err"] = float(d.max())
    ok = (d <= tol + tol * ref.float().abs()).float().mean().item()
    row["matched_ratio"] = ok
    row["pass"] = bool(row["dtype_ok"] and row["shape_ok"] and ok >= 1.0)
    # stability across calls (2nd call already uses the cached launcher)
    g1 = cand.run(x, list(size), ac, sh, sw)
    g2 = cand.run(x, list(size), ac, sh, sw)
    row["stable_vs_first"] = bool(torch.equal(got, g1) and torch.equal(g1, g2))
    # diagnostic timing on the cached path
    for _ in range(3):
        cand.run(x, list(size), ac, sh, sw)
    torch.npu.synchronize()
    samples = []
    for _ in range(5):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            cand.run(x, list(size), ac, sh, sw)
        torch.npu.synchronize()
        samples.append((time.perf_counter() - t0) / 20 * 1e3)
    row["median_ms"] = statistics.median(samples)
    row["ref_median_ms"] = _ref_time(x, size, ac, sh, sw)
    res["rows"].append(row)
    flush()
    return row


def _ref_time(x, size, ac, sh, sw):
    for _ in range(3):
        torch.ops.aten.upsample_bilinear2d(x, list(size), ac, sh, sw)
    torch.npu.synchronize()
    samples = []
    for _ in range(5):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            torch.ops.aten.upsample_bilinear2d(x, list(size), ac, sh, sw)
        torch.npu.synchronize()
        samples.append((time.perf_counter() - t0) / 20 * 1e3)
    return statistics.median(samples)


def mk(shape, dtype, seed=1730):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g).to(dtype=dtype, device="npu")


TIMING = [
    ("t0_fp16", torch.float16, (1, 16, 64, 64), (128, 128)),
    ("t1_fp16", torch.float16, (4, 32, 128, 128), (256, 256)),
    ("t2_bf16", torch.bfloat16, (1, 16, 64, 64), (128, 128)),
    ("t3_bf16", torch.bfloat16, (4, 32, 128, 128), (256, 256)),
    ("t4_fp32", torch.float32, (1, 16, 64, 64), (128, 128)),
    ("t5_fp32", torch.float32, (4, 32, 128, 128), (256, 256)),
]

for name, dtype, shape, size in TIMING:
    try:
        check(name, mk(shape, dtype), size)
    except Exception as exc:
        res["errors"].append(f"{name}: {repr(exc)[:600]}")
        flush()

# fast-path edge cases: odd heights, minimal images, 1-row images
for name, dtype, shape, size in [
    ("odd_7x9_fp32", torch.float32, (1, 3, 7, 9), (14, 18)),
    ("odd_7x8_fp16", torch.float16, (1, 3, 7, 8), (14, 16)),
    ("h4_w8_bf16", torch.bfloat16, (2, 3, 4, 8), (8, 16)),
    ("ih1_fp16", torch.float16, (2, 5, 1, 8), (2, 16)),
    ("ih3_fp32", torch.float32, (1, 1, 3, 8), (6, 16)),
    ("w512_fp16", torch.float16, (1, 2, 16, 512), (32, 1024)),
]:
    try:
        check(name, mk(shape, dtype), size)
    except Exception as exc:
        res["errors"].append(f"{name}: {repr(exc)[:600]}")
        flush()

# fallback domains: strided minor axis, explicit scales, align_corners
try:
    base = mk((1, 8, 32, 128), torch.float16)
    check("strided_w2", base[:, :, :, ::2], (32, 128))
except Exception as exc:
    res["errors"].append(f"strided_w2: {repr(exc)[:400]}")
    flush()
try:
    check("explicit_scales", mk((1, 8, 32, 32), torch.float16), (64, 64), False, 2.0, 2.0)
except Exception as exc:
    res["errors"].append(f"explicit_scales: {repr(exc)[:400]}")
    flush()
try:
    check("align_corners_2x", mk((1, 3, 7, 9), torch.float32), (14, 18), True)
except Exception as exc:
    res["errors"].append(f"align_corners: {repr(exc)[:400]}")
    flush()
try:
    base = mk((1, 8, 64, 32), torch.float16)
    check("transposed_h", base.transpose(2, 3), (128, 64))
except Exception as exc:
    res["errors"].append(f"transposed: {repr(exc)[:400]}")
    flush()

flush()
summary = [(r["case"], r["pass"], r["max_abs_err"], r.get("median_ms"), r.get("ref_median_ms"))
           for r in res["rows"]]
print(json.dumps({"summary": summary, "errors": res["errors"]}, indent=1)[:4000])
