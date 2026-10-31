"""Bounded sweep: tile size and multi-tile load hoisting for log10_ (in-place).

Evidence from the previous job (4096x4096, debug harness):
  mono BLOCK=2048  fp32 307.7 us (8192 programs, 751 ns/program)
  mono BLOCK=8192  fp32  81.1 us (2048 programs, 792 ns/program)
  ATen floor       fp32  27.3 us
Per-program cost is nearly independent of tile size up to 64 KiB, so the cost is
dominated by per-tile fixed latency rather than bandwidth. This sweep asks
whether (i) bigger tiles and (ii) issuing several independent loads before any
store (manual memory-level parallelism) recover the gap.
"""

import json
import os
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
LOG10E = 0.4342944819032518


@triton.jit
def k_mono(x_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs).to(tl.float32)
    tl.store(x_ptr + offs, (tl.log(x) * 0.4342944819032518).to(x_ptr.dtype.element_ty))


@triton.jit
def k_hoist2(x_ptr, SUB: tl.constexpr):
    base = tl.program_id(0) * (2 * SUB)
    o0 = base + tl.arange(0, SUB)
    o1 = o0 + SUB
    a = tl.load(x_ptr + o0).to(tl.float32)
    b = tl.load(x_ptr + o1).to(tl.float32)
    ya = tl.log(a) * 0.4342944819032518
    yb = tl.log(b) * 0.4342944819032518
    tl.store(x_ptr + o0, ya.to(x_ptr.dtype.element_ty))
    tl.store(x_ptr + o1, yb.to(x_ptr.dtype.element_ty))


@triton.jit
def k_hoist4(x_ptr, SUB: tl.constexpr):
    base = tl.program_id(0) * (4 * SUB)
    o0 = base + tl.arange(0, SUB)
    o1 = o0 + SUB
    o2 = o1 + SUB
    o3 = o2 + SUB
    a = tl.load(x_ptr + o0).to(tl.float32)
    b = tl.load(x_ptr + o1).to(tl.float32)
    c = tl.load(x_ptr + o2).to(tl.float32)
    d = tl.load(x_ptr + o3).to(tl.float32)
    ya = tl.log(a) * 0.4342944819032518
    yb = tl.log(b) * 0.4342944819032518
    yc = tl.log(c) * 0.4342944819032518
    yd = tl.log(d) * 0.4342944819032518
    tl.store(x_ptr + o0, ya.to(x_ptr.dtype.element_ty))
    tl.store(x_ptr + o1, yb.to(x_ptr.dtype.element_ty))
    tl.store(x_ptr + o2, yc.to(x_ptr.dtype.element_ty))
    tl.store(x_ptr + o3, yd.to(x_ptr.dtype.element_ty))


def bench(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def make_input(shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g).abs() + 0.05
    return x.to(dtype).npu()


def rel_err(buf, ref):
    a = buf.to(torch.float32)
    b = ref.to(torch.float32)
    return float(((a - b).abs() / b.abs().clamp_min(1e-6)).max())


def run_shape(name, shape, dtype, seed, builds):
    base = make_input(shape, dtype, seed)
    n = base.numel()
    esize = base.element_size()
    ref_out = torch.log10_(base.clone())
    rows = []
    for label, block, kind in builds:
        if n % block:
            rows.append({"variant": label, "skipped": "n % block != 0"})
            continue
        try:
            if kind == "mono":
                grid = (n // block,)
                launch = lambda t, g=grid, b=block: k_mono[g](t, BLOCK=b)
            elif kind == "hoist2":
                grid = (n // (2 * block),)
                launch = lambda t, g=grid, b=block: k_hoist2[g](t, SUB=b)
            else:
                grid = (n // (4 * block),)
                launch = lambda t, g=grid, b=block: k_hoist4[g](t, SUB=b)
            work = base.clone()
            launch(work)
            torch.npu.synchronize()
            row = {"variant": label, "programs": grid[0],
                   "rel_err": rel_err(work, ref_out),
                   "nonfinite": int((~torch.isfinite(work.to(torch.float32))).sum().item())}
            del work
            buf = base.clone()
            row["us"] = bench(lambda: launch(buf))
            row["ns_per_program_core"] = row["us"] * 1000.0 * 20.0 / grid[0] / 20.0
            row["logical_TBs"] = (2.0 * n * esize) / (row["us"] * 1e-6) / 1e12
            del buf
            rows.append(row)
        except Exception as exc:
            rows.append({"variant": label, "error": repr(exc)[-300:]})

    ref_buf = base.clone()
    ref_us = bench(lambda: torch.log10_(ref_buf))
    return {"workload": name, "dtype": str(dtype), "n": n, "ref_us": ref_us,
            "ref_logical_TBs": (2.0 * n * esize) / (ref_us * 1e-6) / 1e12,
            "variants": rows}


FP32_BUILDS = [
    ("mono_8192", 8192, "mono"),
    ("mono_16384", 16384, "mono"),
    ("mono_32768", 32768, "mono"),
    ("hoist2_8192", 8192, "hoist2"),
    ("hoist4_4096", 4096, "hoist4"),
    ("hoist2_16384", 16384, "hoist2"),
]

FP16_BUILDS = [
    ("mono_16384", 16384, "mono"),
    ("mono_32768", 32768, "mono"),
    ("mono_65536", 65536, "mono"),
    ("hoist2_16384", 16384, "hoist2"),
    ("hoist4_8192", 8192, "hoist4"),
    ("hoist2_32768", 32768, "hoist2"),
]


def main():
    out = {"cases": []}
    out["cases"].append(run_shape("fp32_4096x4096", (4096, 4096), torch.float32,
                                  1740, FP32_BUILDS))
    out["cases"].append(run_shape("fp16_4096x4096", (4096, 4096), torch.float16,
                                  1732, FP16_BUILDS))
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "log10_tiles.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
