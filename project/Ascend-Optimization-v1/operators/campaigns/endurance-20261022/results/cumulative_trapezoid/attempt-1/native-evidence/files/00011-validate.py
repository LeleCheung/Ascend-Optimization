"""Validate and time the candidate kernel (tmp/main.py) against torch on the
authoritative workload shapes plus layout/precision edge cases.
"""
import importlib.util
import json
import os
import time

import torch
import triton

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
res = {"checks": [], "timings": [], "errors": [], "info": {}}

spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)

TOL = {torch.float16: (0.015, 0.015), torch.bfloat16: (0.015, 0.015), torch.float32: (1e-4, 1e-4)}


def make(shape, dtype, seed, transpose=False, dim=-1):
    y = torch.randn(shape, generator=torch.Generator().manual_seed(seed)).to(dtype).npu()
    if transpose:
        y = y.transpose(-1, -2)
    return y


def check(name, y, dx, dim):
    try:
        ref = torch.cumulative_trapezoid(y, dx=dx, dim=dim)
        out = cand.run(y, dx=dx, dim=dim)
        assert out.dtype == ref.dtype, f"dtype {out.dtype} != {ref.dtype}"
        assert tuple(out.shape) == tuple(ref.shape), f"shape {tuple(out.shape)} != {tuple(ref.shape)}"
        if ref.numel() == 0:
            res["checks"].append({"case": name, "ok": True, "empty": True})
            return
        atol, rtol = TOL[ref.dtype]
        a = out.float()
        b = ref.float()
        diff = (a - b).abs()
        tol = atol + rtol * b.abs()
        bad = (diff > tol).sum().item()
        res["checks"].append({
            "case": name, "ok": bad == 0, "bad": bad, "n": ref.numel(),
            "maxabs": diff.max().item(),
            "maxrel": (diff / b.abs().clamp_min(1e-9)).max().item(),
        })
    except Exception as e:
        res["errors"].append({"case": name, "err": repr(e)[-400:]})


def wall(fn, iters=30, warm=8):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def dev(fn, iters=30, warm=8):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    e0 = torch.npu.Event(enable_timing=True)
    e1 = torch.npu.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.npu.synchronize()
    return e0.elapsed_time(e1) / iters


# ---- authoritative correctness specs (name, spec, seed)
CORR = [
    ("c0", dict(dtype="float16", shape=[3, 17], dim=-1, dx=1, transpose=False), 1729),
    ("c1", dict(dtype="float16", shape=[3, 17], dim=-1, dx=0.25, transpose=False), 1730),
    ("c2", dict(dtype="float16", shape=[7, 9], dim=0, dx=1, transpose=False), 1731),
    ("c3", dict(dtype="float16", shape=[7, 9], dim=0, dx=0.25, transpose=False), 1732),
    ("c4", dict(dtype="float16", shape=[7, 9], dim=-1, dx=1, transpose=True), 1733),
    ("c5", dict(dtype="float16", shape=[7, 9], dim=-1, dx=0.25, transpose=True), 1734),
    ("c6", dict(dtype="float16", shape=[3, 1], dim=-1, dx=1, transpose=False), 1735),
    ("c7", dict(dtype="float16", shape=[3, 1], dim=-1, dx=0.25, transpose=False), 1736),
    ("c8", dict(dtype="bfloat16", shape=[3, 17], dim=-1, dx=1, transpose=False), 1737),
    ("c9", dict(dtype="bfloat16", shape=[3, 17], dim=-1, dx=0.25, transpose=False), 1738),
    ("c10", dict(dtype="bfloat16", shape=[7, 9], dim=0, dx=1, transpose=False), 1739),
    ("c11", dict(dtype="bfloat16", shape=[7, 9], dim=0, dx=0.25, transpose=False), 1740),
    ("c12", dict(dtype="bfloat16", shape=[7, 9], dim=-1, dx=1, transpose=True), 1741),
    ("c13", dict(dtype="bfloat16", shape=[7, 9], dim=-1, dx=0.25, transpose=True), 1742),
    ("c14", dict(dtype="bfloat16", shape=[3, 1], dim=-1, dx=1, transpose=False), 1743),
    ("c15", dict(dtype="bfloat16", shape=[3, 1], dim=-1, dx=0.25, transpose=False), 1744),
    ("c16", dict(dtype="float32", shape=[3, 17], dim=-1, dx=1, transpose=False), 1745),
    ("c17", dict(dtype="float32", shape=[3, 17], dim=-1, dx=0.25, transpose=False), 1746),
    ("c18", dict(dtype="float32", shape=[7, 9], dim=0, dx=1, transpose=False), 1747),
    ("c19", dict(dtype="float32", shape=[7, 9], dim=0, dx=0.25, transpose=False), 1748),
    ("c20", dict(dtype="float32", shape=[7, 9], dim=-1, dx=1, transpose=True), 1749),
    ("c21", dict(dtype="float32", shape=[7, 9], dim=-1, dx=0.25, transpose=True), 1750),
    ("c22", dict(dtype="float32", shape=[3, 1], dim=-1, dx=1, transpose=False), 1751),
    ("c23", dict(dtype="float32", shape=[3, 1], dim=-1, dx=0.25, transpose=False), 1752),
]
TIM = [
    ("t0", dict(dtype="float16", shape=[128, 1024], dim=-1, dx=1), 1729),
    ("t1", dict(dtype="float16", shape=[1024, 4096], dim=-1, dx=1), 1730),
    ("t2", dict(dtype="float16", shape=[1024, 128], dim=0, dx=1), 1731),
    ("t3", dict(dtype="bfloat16", shape=[128, 1024], dim=-1, dx=1), 1732),
    ("t4", dict(dtype="bfloat16", shape=[1024, 4096], dim=-1, dx=1), 1733),
    ("t5", dict(dtype="bfloat16", shape=[1024, 128], dim=0, dx=1), 1734),
    ("t6", dict(dtype="float32", shape=[128, 1024], dim=-1, dx=1), 1735),
    ("t7", dict(dtype="float32", shape=[1024, 4096], dim=-1, dx=1), 1736),
    ("t8", dict(dtype="float32", shape=[1024, 128], dim=0, dx=1), 1737),
]
EXTRA = [
    ("3d-mid", dict(dtype="float32", shape=[4, 33, 8], dim=1, dx=1.0), 100),
    ("3d-last", dict(dtype="bfloat16", shape=[4, 5, 300], dim=-1, dx=1.0), 101),
    ("1d", dict(dtype="float32", shape=[257], dim=-1, dx=0.5), 102),
    ("n33", dict(dtype="float16", shape=[40, 33], dim=-1, dx=1.0), 103),
    ("n34", dict(dtype="float32", shape=[40, 34], dim=-1, dx=1.0), 104),
    ("n129", dict(dtype="float32", shape=[40, 129], dim=-1, dx=1.0), 105),
    ("big-neg-dim", dict(dtype="float16", shape=[8, 300], dim=-2, dx=1.0), 106),
]


def build(s, seed):
    tdt = getattr(torch, s["dtype"])
    return make(s["shape"], tdt, seed, s.get("transpose", False), s["dim"])


def main():
    torch.npu.set_device(0)
    for name, s, seed in CORR + EXTRA:
        check(name, build(s, seed), s["dx"], s["dim"])
    # large-N numerics sanity (not gated by the formal correctness set)
    for dt in ("float16", "bfloat16", "float32"):
        tdt = getattr(torch, dt)
        y = make([1024, 4096], tdt, 1730)
        check(f"large-{dt}", y, 1, -1)

    for name, s, seed in TIM:
        y = build(s, seed)
        dx, dim = s["dx"], s["dim"]
        ref = torch.cumulative_trapezoid(y, dx=dx, dim=dim)
        cand.run(y, dx=dx, dim=dim)
        out = cand.run(y, dx=dx, dim=dim)          # second call: prebuilt launcher
        bad = ((out.float() - ref.float()).abs() > 0.015 + 0.015 * ref.float().abs()).sum().item()
        plan = cand._PLANS.get((y.dtype, y.shape, y.stride(), y.dim(), dim, float(dx)))
        rec = {"case": name, "n_bad": bad, "fast_launcher": bool(plan and plan[1]),
               "wall_ms": wall(lambda: cand.run(y, dx=dx, dim=dim)),
               "dev_ms": dev(lambda: cand.run(y, dx=dx, dim=dim)),
               "torch_wall_ms": wall(lambda: torch.cumulative_trapezoid(y, dx=dx, dim=dim))}
        if plan and plan[1] is not None:
            outb = y.new_empty(plan[0])
            rec["fire_only_ms"] = wall(lambda: plan[1](y, outb))
            rec["alloc_ms"] = wall(lambda: y.new_empty(plan[0]))
        res["timings"].append(rec)

    with open(os.path.join(ART, "validate.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1)[:26000])


if __name__ == "__main__":
    main()
