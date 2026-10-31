"""Bounded diagnostic: seed kernel numerics vs torch.native_group_norm + dtype contract."""
import json
import os
import sys
import time

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.getcwd())
import importlib.util

spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)


CASES = [
    (torch.float16, (1, 8, 3, 5), 4, True),
    (torch.float16, (2, 32, 1, 1), 4, True),
    (torch.bfloat16, (2, 16, 7, 9), 4, True),
    (torch.float32, (1, 8, 3, 5), 4, False),
    (torch.float32, (2, 32, 1, 1), 4, False),
    (torch.float16, (4, 32, 32, 32), 8, True),
    (torch.float16, (16, 128, 32, 32), 8, True),
    (torch.bfloat16, (8, 64, 64, 64), 8, True),
    (torch.float32, (4, 32, 32, 32), 8, True),
    (torch.float32, (16, 128, 32, 32), 8, True),
]


def tol(dt):
    return 1e-4 if dt == torch.float32 else 1.5e-2


def match_ratio(a, b, atol, rtol):
    a = a.float()
    b = b.float()
    ok = (a - b).abs() <= (atol + rtol * b.abs())
    return ok.float().mean().item(), (a - b).abs().max().item()


out_records = []


def timeit(fn, iters=50):
    for _ in range(10):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


for dt, shape, groups, affine in CASES:
    gen = torch.Generator().manual_seed(1729)
    x = torch.randn(shape, generator=gen).to(dtype=dt, device="npu")
    c = shape[1]
    w = (torch.randn([c], generator=gen).abs() + 0.05).to(dtype=dt, device="npu") if affine else None
    b = (torch.randn([c], generator=gen).abs() + 0.05).to(dtype=dt, device="npu") if affine else None
    n, cx = shape[:2]
    hw = 1
    for d in shape[2:]:
        hw *= d

    ro, rm, rr = torch.native_group_norm(x, w, b, n, cx, hw, groups, 1e-5)
    co, cm, cr = cand.run(x, w, b, n, cx, hw, groups, 1e-5)
    torch.npu.synchronize()

    at, rt = tol(dt)
    rec = {
        "dtype": str(dt),
        "shape": list(shape),
        "groups": groups,
        "affine": affine,
        "ref_dtypes": [str(ro.dtype), str(rm.dtype), str(rr.dtype)],
        "cand_dtypes": [str(co.dtype), str(cm.dtype), str(cr.dtype)],
        "ref_shapes": [list(ro.shape), list(rm.shape), list(rr.shape)],
        "out": match_ratio(co, ro, at, rt),
        "mean": match_ratio(cm, rm, at, rt),
        "rstd": match_ratio(cr, rr, at, rt),
    }
    # strided (last-dim stride 2) variant for the same geometry
    if len(shape) >= 3:
        base = torch.empty((*shape, 2), dtype=dt, device="npu")
        base[..., 0] = x
        xs = base[..., 0]
        so, sm, sr = torch.native_group_norm(xs, w, b, n, cx, hw, groups, 1e-5)
        sco, scm, scr = cand.run(xs, w, b, n, cx, hw, groups, 1e-5)
        torch.npu.synchronize()
        rec["strided_out"] = match_ratio(sco, so, at, rt)
        rec["strided_mean"] = match_ratio(scm, sm, at, rt)

    try:
        t_ref = timeit(lambda: torch.native_group_norm(x, w, b, n, cx, hw, groups, 1e-5))
        t_cand = timeit(lambda: cand.run(x, w, b, n, cx, hw, groups, 1e-5))
        rec["ref_us"] = t_ref
        rec["cand_us"] = t_cand
    except Exception as exc:  # noqa: BLE001
        rec["timing_error"] = repr(exc)
    out_records.append(rec)
    print(json.dumps(rec), flush=True)

art = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
os.makedirs(art, exist_ok=True)
with open(os.path.join(art, "gn_probe.json"), "w") as f:
    json.dump(out_records, f, indent=1)
print("DONE", len(out_records))
