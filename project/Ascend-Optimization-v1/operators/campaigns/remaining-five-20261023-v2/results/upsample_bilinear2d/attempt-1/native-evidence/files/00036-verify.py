"""Numeric verification on all frozen workload specs + row-tile sweep.

The native timing phase does not compare outputs, so the timing fixtures are
verified here explicitly against a float64 CPU reference under the contract
tolerance.  Timings use a 50-iteration loop between two events so the per-call
host submission cost is amortised (the per-iteration event span on this part
has a ~0.1 ms floor that hides everything at these sizes).
"""

import json
import os
import statistics
import sys
import traceback

import torch
import torch_npu  # noqa: F401

import triton

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import main as cand  # noqa: E402

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
DEV = "npu:0"

SPECS = [
    # (uuid, dtype, shape, output_size, align_corners, seed)
    ("c0", torch.float16, (1, 3, 7, 9), (13, 17), True, 1729),
    ("c1", torch.float16, (1, 3, 7, 9), (13, 17), False, 1730),
    ("c2", torch.float16, (1, 3, 7, 9), (1, 1), True, 1731),
    ("c3", torch.float16, (1, 3, 7, 9), (1, 1), False, 1732),
    ("c4", torch.float16, (1, 3, 7, 9), (3, 5), True, 1733),
    ("c5", torch.float16, (1, 3, 7, 9), (3, 5), False, 1734),
    ("c6", torch.bfloat16, (1, 3, 7, 9), (13, 17), True, 1735),
    ("c7", torch.bfloat16, (1, 3, 7, 9), (13, 17), False, 1736),
    ("c8", torch.bfloat16, (1, 3, 7, 9), (1, 1), True, 1737),
    ("c9", torch.bfloat16, (1, 3, 7, 9), (1, 1), False, 1738),
    ("c10", torch.bfloat16, (1, 3, 7, 9), (3, 5), True, 1739),
    ("c11", torch.bfloat16, (1, 3, 7, 9), (3, 5), False, 1740),
    ("c12", torch.float32, (1, 3, 7, 9), (13, 17), True, 1741),
    ("c13", torch.float32, (1, 3, 7, 9), (13, 17), False, 1742),
    ("c14", torch.float32, (1, 3, 7, 9), (1, 1), True, 1743),
    ("c15", torch.float32, (1, 3, 7, 9), (1, 1), False, 1744),
    ("c16", torch.float32, (1, 3, 7, 9), (3, 5), True, 1745),
    ("c17", torch.float32, (1, 3, 7, 9), (3, 5), False, 1746),
    ("t0", torch.float16, (1, 16, 64, 64), (128, 128), False, 1729),
    ("t1", torch.float16, (4, 32, 128, 128), (256, 256), False, 1730),
    ("t2", torch.bfloat16, (1, 16, 64, 64), (128, 128), False, 1731),
    ("t3", torch.bfloat16, (4, 32, 128, 128), (256, 256), False, 1732),
    ("t4", torch.float32, (1, 16, 64, 64), (128, 128), False, 1733),
    ("t5", torch.float32, (4, 32, 128, 128), (256, 256), False, 1734),
]


def tol_for(dtype):
    if dtype == torch.float32:
        return 1e-4, 1e-4
    return 0.015, 0.015


def tolerance_check(got, ref_f64, atol, rtol):
    a = got.to(torch.float32).double()
    b = ref_f64
    ok = torch.abs(a - b) <= atol + rtol * torch.abs(b)
    return {
        "matched_ratio": float(ok.double().mean().item()),
        "max_abs": float(torch.max(torch.abs(a - b)).item()),
    }


def main():
    out = {"numerics": {}, "timings": {}}
    torch.npu.set_device(DEV)

    for name, dt, shape, osz, ac, seed in SPECS:
        try:
            g = torch.Generator().manual_seed(seed)
            x_cpu = torch.randn(shape, generator=g).to(dtype=dt)
            ref = torch.nn.functional.interpolate(
                x_cpu.to(torch.float64), size=tuple(osz),
                mode="bilinear", align_corners=ac)
            x = x_cpu.to(DEV)
            y = cand.run(x, list(osz), ac)
            atol, rtol = tol_for(dt)
            st = tolerance_check(y.cpu(), ref, atol, rtol)
            st["dtype_ok"] = str(y.dtype) == str(x.dtype)
            st["shape_ok"] = tuple(y.shape) == (shape[0], shape[1], osz[0], osz[1])
            st["input_intact"] = bool(torch.equal(x.cpu(), x_cpu))
            st["alloc_new"] = y.data_ptr() != x.data_ptr()
            out["numerics"][name] = st
        except Exception:
            out["numerics"][name] = "ERR:" + traceback.format_exc().splitlines()[-1][:220]

    # row-tile sweep on the two timing geometries
    def bench_call(fn, iters=50):
        for _ in range(5):
            fn()
        torch.npu.synchronize()
        e0 = torch.npu.Event(enable_timing=True)
        e1 = torch.npu.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.npu.synchronize()
        return round(e0.elapsed_time(e1) / iters, 5)

    for name, dt, shape, osz, ac, seed in SPECS[18:]:
        N, C, IH, IW = shape
        OH, OW = osz
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(shape, generator=g).to(dtype=dt, device=DEV)
        n_rows = N * C * OH
        iwp = triton.next_power_of_2(IW)
        r = {}
        for br in (4, 8, 16, 32):
            if br > iwp * 8 or n_rows % br != 0:
                continue
            outp = torch.empty((N, C, OH, OW), device=DEV, dtype=dt)
            grid = (n_rows // br,)
            try:
                r[f"BR{br}_prog{n_rows // br}"] = bench_call(
                    lambda br=br, outp=outp, grid=grid: cand._bilinear_int_kernel[grid](
                        outp, x, n_rows, x.stride(0), x.stride(1), x.stride(2),
                        IH / OH, C=C, IH=IH, IW=IW, OH=OH, OW=OW, BR=br, IWP=iwp))
            except Exception:
                r[f"BR{br}"] = "ERR:" + traceback.format_exc().splitlines()[-1][:160]
        # full run() path (includes allocation) for the currently chosen tile
        r["full_run"] = bench_call(lambda: cand.run(x, list(osz), ac))
        out["timings"][name] = r

    with open(os.path.join(ART, "verify.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


main()
