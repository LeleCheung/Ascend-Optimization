"""Probe tile-width / math-width combinations for in-place SELU on Ascend.

Prior evidence: device time scales with the NUMBER of MTE transfers, not with the
program count (copy b16384 40us vs b8192 78us; 2 tiles/program at b8192 = 77us).
So the lever is the largest transfer that fits in UB, and the UB pressure is set
by how many full-width tiles the math needs live.

This probe reports, per (dtype, BLOCK, math width, ubuf saving) combination:
  - whether it compiles (UB overflow is a compile-time error)
  - device time at n = 16777216 through the cheap raw launcher
  - numeric error vs the PyTorch reference screen against the case tolerance
"""
import json
import os
import time
import traceback

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
OUT = {}


# --------------------------------------------------------------------- kernels
@triton.jit
def k_f32(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    xf = x.to(tl.float32)
    y = 1.0507009873554805 * tl.where(xf > 0, xf, 1.6732632423543772 * (tl.exp(xf) - 1.0))
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_f32_fma(X, BLOCK: tl.constexpr):
    """One fewer full-width temporary: dropout of the separate *1.6733 pass."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    xf = x.to(tl.float32)
    n = 1.7580993407452793 * tl.exp(xf) - 1.7580993407452793
    y = tl.where(xf > 0, 1.0507009873554805 * xf, n)
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_nat(X, BLOCK: tl.constexpr):
    """All math in the input dtype (no full-width fp32 tile)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * (tl.exp(x) - 1.0))
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_exp32(X, BLOCK: tl.constexpr):
    """exp and the -1 cancellation in fp32, coefficient multiply + select narrow."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    xf = x.to(tl.float32)
    n = (tl.exp(xf) - 1.0).to(X.dtype.element_ty)
    n = 1.7580993407452793 * n
    y = tl.where(x > 0, 1.0507009873554805 * x, n)
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_copy(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(X + offs, tl.load(X + offs))


_RT = {}


def stream_getter():
    if "g" not in _RT:
        dev = torch.npu.current_device()
        raw = torch_npu._C._npu_getCurrentRawStreamNoWait
        _RT["g"] = lambda: raw(dev)
    return _RT["g"]


def bench(fn, iters=50, warm=10):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    host_us = (time.perf_counter() - t0) / iters * 1e6
    torch.npu.synchronize()
    total_us = (time.perf_counter() - t0) / iters * 1e6
    return {"total_us": round(total_us, 2), "host_us": round(host_us, 2)}


def run_variant(kern, x, block, n, opts):
    grid = n // block
    compiled = kern.warmup(x, BLOCK=block, grid=(1,), **opts)
    launcher = compiled.run
    call = launcher.launch
    fn_h = compiled.function
    meta = compiled.packed_metadata
    get = stream_getter()

    def invoke():
        call(grid, 1, 1, get(), fn_h, meta, None, None, None, x, None)

    invoke()
    torch.npu.synchronize()
    return bench(invoke)


def main():
    n = 4096 * 4096
    OUT["n"] = n
    tol = {"torch.float16": 0.015, "torch.bfloat16": 0.015, "torch.float32": 0.0001}

    plan = {
        torch.float16: [
            ("copy_b16384", k_copy, 16384, {}),
            ("nat_b16384", k_nat, 16384, {}),
        ],
        torch.bfloat16: [
            ("f32_b8192", k_f32, 8192, {}),
            ("f32_b16384", k_f32, 16384, {}),
            ("f32_b16384_ub", k_f32, 16384, {"enable_ubuf_saving": True}),
            ("fma_b16384_ub", k_f32_fma, 16384, {"enable_ubuf_saving": True}),
            ("nat_b16384", k_nat, 16384, {}),
            ("nat_b32768", k_nat, 32768, {}),
            ("nat_b32768_ub", k_nat, 32768, {"enable_ubuf_saving": True}),
            ("exp32_b16384", k_exp32, 16384, {}),
            ("exp32_b32768", k_exp32, 32768, {}),
            ("copy_b16384", k_copy, 16384, {}),
            ("copy_b32768", k_copy, 32768, {}),
        ],
        torch.float32: [
            ("f32_b8192", k_f32, 8192, {}),
            ("fma_b16384_ub", k_f32_fma, 16384, {"enable_ubuf_saving": True}),
            ("fma_b16384", k_f32_fma, 16384, {}),
            ("copy_b16384", k_copy, 16384, {}),
            ("copy_b32768", k_copy, 32768, {}),
        ],
    }

    for dtype, variants in plan.items():
        tag = str(dtype).replace("torch.", "")
        rows = {}
        base = torch.randn(n, dtype=dtype, device="npu:0")
        ref = torch.nn.functional.selu(base.clone().float()).to(dtype)
        for name, kern, block, opts in variants:
            row = {"block": block, "opts": opts or None}
            try:
                x = base.clone()
                row.update(run_variant(kern, x, block, n, opts))
                if kern is not k_copy:
                    torch.npu.synchronize()
                    d = (x.float() - ref.float()).abs()
                    bad = (d > tol[str(dtype)] + tol[str(dtype)] * ref.float().abs()).sum().item()
                    row["max_abs_err"] = round(d.max().item(), 8)
                    row["bad_elems"] = int(bad)
                del x
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).replace("\n", " ")
                row["error"] = msg[-260:]
            rows[name] = row
            print(tag, name, row, flush=True)
        OUT[tag] = rows
        del base, ref

    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "tile_probe.json"), "w") as fh:
        json.dump(OUT, fh, indent=2, default=str)
    print(json.dumps(OUT, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
