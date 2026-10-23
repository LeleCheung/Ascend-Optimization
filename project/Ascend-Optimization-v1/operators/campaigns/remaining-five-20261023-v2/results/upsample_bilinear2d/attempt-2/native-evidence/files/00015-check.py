"""Validate and time the current candidate on every frozen workload input.

Diagnostic only: builds inputs exactly like the harness gen_inputs, compares
against the PyTorch reference, and reports walltime for the timing cases.
"""

import json
import os
import sys
import time
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tmp"))
import main as cand  # noqa: E402

RES = {}


def rec(k, v):
    RES[k] = v
    print("%s = %s" % (k, v), flush=True)


def adir():
    d = os.environ.get("KGS_DEBUG_ARTIFACTS", "")
    return d if d and os.path.isdir(d) else os.getcwd()


DEV = "npu:0"

# (name, dtype, shape, out, align_corners, seed, atol, rtol)
CORRECT = [
    ("c0", "float16", [1, 3, 7, 9], [13, 17], True, 1729, 0.015, 0.015),
    ("c1", "float16", [1, 3, 7, 9], [13, 17], False, 1730, 0.015, 0.015),
    ("c2", "float16", [1, 3, 7, 9], [1, 1], True, 1731, 0.015, 0.015),
    ("c3", "float16", [1, 3, 7, 9], [1, 1], False, 1732, 0.015, 0.015),
    ("c4", "float16", [1, 3, 7, 9], [3, 5], True, 1733, 0.015, 0.015),
    ("c5", "float16", [1, 3, 7, 9], [3, 5], False, 1734, 0.015, 0.015),
    ("c6", "bfloat16", [1, 3, 7, 9], [13, 17], True, 1735, 0.015, 0.015),
    ("c7", "bfloat16", [1, 3, 7, 9], [13, 17], False, 1736, 0.015, 0.015),
    ("c8", "bfloat16", [1, 3, 7, 9], [1, 1], True, 1737, 0.015, 0.015),
    ("c9", "bfloat16", [1, 3, 7, 9], [1, 1], False, 1738, 0.015, 0.015),
    ("c10", "bfloat16", [1, 3, 7, 9], [3, 5], True, 1739, 0.015, 0.015),
    ("c11", "bfloat16", [1, 3, 7, 9], [3, 5], False, 1740, 0.015, 0.015),
    ("c12", "float32", [1, 3, 7, 9], [13, 17], True, 1741, 0.0001, 0.0001),
    ("c13", "float32", [1, 3, 7, 9], [13, 17], False, 1742, 0.0001, 0.0001),
    ("c14", "float32", [1, 3, 7, 9], [1, 1], True, 1743, 0.0001, 0.0001),
    ("c15", "float32", [1, 3, 7, 9], [1, 1], False, 1744, 0.0001, 0.0001),
    ("c16", "float32", [1, 3, 7, 9], [3, 5], True, 1745, 0.0001, 0.0001),
    ("c17", "float32", [1, 3, 7, 9], [3, 5], False, 1746, 0.0001, 0.0001),
]

TIMING = [
    ("t0", "float16", [1, 16, 64, 64], [128, 128], False, 1729, 0.015, 0.015),
    ("t1", "float16", [4, 32, 128, 128], [256, 256], False, 1730, 0.015, 0.015),
    ("t2", "bfloat16", [1, 16, 64, 64], [128, 128], False, 1731, 0.015, 0.015),
    ("t3", "bfloat16", [4, 32, 128, 128], [256, 256], False, 1732, 0.015, 0.015),
    ("t4", "float32", [1, 16, 64, 64], [128, 128], False, 1733, 0.0001, 0.0001),
    ("t5", "float32", [4, 32, 128, 128], [256, 256], False, 1734, 0.0001, 0.0001),
]


def gen(dtype, shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g).to(dtype=getattr(torch, dtype), device=DEV)


def check(tag, x, oh, ow, ac, atol, rtol):
    out = cand.run(x, [oh, ow], ac, None, None)
    ref = torch.ops.aten.upsample_bilinear2d(x, [oh, ow], ac, None, None)
    if out.shape != ref.shape:
        rec("shape_" + tag, "%s vs %s" % (tuple(out.shape), tuple(ref.shape)))
        return
    if out.dtype != ref.dtype:
        rec("dtype_" + tag, "%s vs %s" % (out.dtype, ref.dtype))
        return
    a = out.float()
    b = ref.float()
    d = (a - b).abs()
    ok = d <= (atol + rtol * b.abs())
    rec("maxdiff_" + tag, d.max().item())
    rec("matched_" + tag, ok.float().mean().item())
    if out.data_ptr() == x.data_ptr():
        rec("alias_" + tag, "ALIASED")


def bench(tag, x, oh, ow, ac, iters=20):
    def call():
        return cand.run(x, [oh, ow], ac, None, None)

    def refc():
        return torch.ops.aten.upsample_bilinear2d(x, [oh, ow], ac, None, None)

    for f in (call, refc):
        for _ in range(5):
            f()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        call()
    torch.npu.synchronize()
    tc = (time.perf_counter() - t0) / iters * 1e3
    t0 = time.perf_counter()
    for _ in range(iters):
        refc()
    torch.npu.synchronize()
    tr = (time.perf_counter() - t0) / iters * 1e3
    rec("wall_" + tag, tc)
    rec("ref_" + tag, tr)
    rec("speedup_" + tag, (tr / tc) if tc > 0 else 0.0)


def main():
    for spec in CORRECT:
        tag, dt, shp, o, ac, seed, atol, rtol = spec
        try:
            x = gen(dt, shp, seed)
            check(tag, x, o[0], o[1], ac, atol, rtol)
        except Exception:
            rec("err_" + tag, traceback.format_exc()[-600:])
    for spec in TIMING:
        tag, dt, shp, o, ac, seed, atol, rtol = spec
        try:
            x = gen(dt, shp, seed)
            check("num_" + tag, x, o[0], o[1], ac, atol, rtol)
            bench(tag, x, o[0], o[1], ac)
        except Exception:
            rec("err_" + tag, traceback.format_exc()[-600:])
    # extra: non-contiguous and odd geometries must not crash
    try:
        base = torch.randn((1, 3, 7, 18), device=DEV, dtype=torch.float16)
        xs = base[..., ::2]
        check("stride2", xs, 13, 17, False, 0.015, 0.015)
    except Exception:
        rec("err_stride2", traceback.format_exc()[-600:])
    try:
        x = gen("float32", [2, 5, 33, 17], 99)
        check("odd2x", x, 66, 34, False, 0.0001, 0.0001)
    except Exception:
        rec("err_odd2x", traceback.format_exc()[-600:])


try:
    main()
except Exception:
    rec("fatal", traceback.format_exc())

p = os.path.join(adir(), "check.json")
with open(p, "w") as fh:
    json.dump(RES, fh, indent=2, default=str)
print("wrote", p, flush=True)
