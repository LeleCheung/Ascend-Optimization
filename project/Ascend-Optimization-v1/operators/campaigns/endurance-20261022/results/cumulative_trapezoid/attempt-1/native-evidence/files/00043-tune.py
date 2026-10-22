"""Sweep dot-tile choices on the timing shapes, verify numerics, and measure a
pre-bound launcher for the candidate kernel in tmp/main.py.
"""
import importlib.util
import json
import os
import time

import torch
import triton

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
res = {"cases": [], "errors": [], "info": {}}

spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)


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


def main():
    torch.npu.set_device(0)
    shapes = [(128, 1024, 1729), (1024, 4096, 1730)]
    for R, N, seed in shapes:
        for dt in ("float16", "bfloat16", "float32"):
            tdt = getattr(torch, dt)
            y = torch.randn(R, N, generator=torch.Generator().manual_seed(seed)).to(tdt).npu()
            ref = torch.cumulative_trapezoid(y, dx=1.0, dim=-1)
            atol = 0.015 if dt != "float32" else 1e-4
            res["cases"].append({"case": f"torchREF-{dt}-{R}x{N}",
                                 "wall_ms": wall(lambda: torch.cumulative_trapezoid(y, dx=1.0, dim=-1))})
            for BR, L in ((16, 64), (16, 128), (32, 64), (32, 128), (64, 64)):
                if R < BR:
                    continue
                out = torch.empty(R, N - 1, dtype=tdt, device="npu:0")
                g = (triton.cdiv(R, BR),)

                def fn(BR=BR, L=L, out=out, y=y, g=g, R=R, N=N):
                    cand._dot_scan_kernel[g](
                        y, out, N=N, NOUT=N - 1, ROWS=R, ROW_IN=N, SCAN_IN=1,
                        MASK_ROWS=(R % BR) != 0, BR=BR, L=L, SCALE=0.5)
                name = f"dot-{dt}-{R}x{N}-br{BR}-L{L}"
                try:
                    fn()
                    torch.npu.synchronize()
                    d = (out.float() - ref.float()).abs()
                    bad = (d > atol + atol * ref.float().abs()).sum().item()
                    res["cases"].append({"case": name, "wall_ms": wall(fn), "dev_ms": dev(fn),
                                         "maxabs": d.max().item(), "bad": bad})
                except Exception as e:
                    res["errors"].append({"case": name, "err": repr(e)[-260:]})
            # pre-bound runner with the default tile
            try:
                br, L = cand._rows_tile(N - 1, R)
                out = torch.empty(R, N - 1, dtype=tdt, device="npu:0")
                g = (triton.cdiv(R, br),)
                ck = cand._dot_scan_kernel.warmup(
                    y, out, N=N, NOUT=N - 1, ROWS=R, ROW_IN=N, SCAN_IN=1,
                    MASK_ROWS=(R % br) != 0, BR=br, L=L, SCALE=0.5, grid=g)
                runner = ck[(g[0], 1, 1)]
                runner(y, out)
                torch.npu.synchronize()
                d = (out.float() - ref.float()).abs().max().item()
                res["cases"].append({"case": f"runner-{dt}-{R}x{N}",
                                     "wall_ms": wall(lambda: runner(y, out)),
                                     "dev_ms": dev(lambda: runner(y, out)), "maxabs": d})
            except Exception as e:
                res["errors"].append({"case": f"runner-{dt}-{R}x{N}", "err": repr(e)[-400:]})

    with open(os.path.join(ART, "tune.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1)[:24000])


if __name__ == "__main__":
    main()
