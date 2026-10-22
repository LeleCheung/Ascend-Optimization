"""Bounded target diagnostic for in-place SELU launch geometry.

Answers three questions with measurements on the real target:
  1. Which device-property keys are available (num_vectorcore / num_aicore)?
  2. Is the 4096x4096 cost per-program-dispatch bound or vector-instruction bound?
     -> compare math kernel vs pure copy kernel across tile sizes / grid shapes.
  3. Does a grid-stride persistent loop collapse that cost?

Every variant is individually guarded so one UB overflow does not abort the run.
Writes machine-readable results to $KGS_DEBUG_ARTIFACTS/selu_geom.json
"""

import json
import os
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
OUT = {}


def props():
    from triton.runtime import driver

    dev = torch.npu.current_device()
    info = {"current_device": dev}
    try:
        p = driver.active.utils.get_device_properties(dev)
        info["get_device_properties"] = p if isinstance(p, dict) else dict(p)
    except Exception as exc:  # noqa: BLE001
        info["get_device_properties_error"] = repr(exc)
    try:
        info["npu_device_name"] = torch.npu.get_device_name(dev)
    except Exception as exc:  # noqa: BLE001
        info["npu_device_name_error"] = repr(exc)
    try:
        info["npu_device_properties"] = str(torch.npu.get_device_properties(dev))
    except Exception as exc:  # noqa: BLE001
        info["npu_device_properties_error"] = repr(exc)
    try:
        info["raw_stream_available"] = hasattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait")
    except Exception as exc:  # noqa: BLE001
        info["raw_stream_available"] = repr(exc)
    return info


@triton.jit
def k_selu(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    xf = x.to(tl.float32)
    y = 1.0507009873554805 * tl.where(xf > 0, xf, 1.6732632423543772 * (tl.exp(xf) - 1.0))
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_copy(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(X + offs, tl.load(X + offs))


@triton.jit
def k_selu_gs(X, NT, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    g = tl.num_programs(0)
    for i in tl.range(0, NT):
        t = pid + i * g
        offs = t * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        xf = x.to(tl.float32)
        y = 1.0507009873554805 * tl.where(xf > 0, xf, 1.6732632423543772 * (tl.exp(xf) - 1.0))
        tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_copy_gs(X, NT, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    g = tl.num_programs(0)
    for i in tl.range(0, NT):
        t = pid + i * g
        offs = t * BLOCK + tl.arange(0, BLOCK)
        tl.store(X + offs, tl.load(X + offs))


def bench(fn, iters=40, warmup=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    host_us = (time.perf_counter() - t0) / iters * 1e6
    torch.npu.synchronize()
    total_us = (time.perf_counter() - t0) / iters * 1e6
    return {"total_us": round(total_us, 3), "host_us": round(host_us, 3)}


def safe(fn):
    try:
        return bench(fn)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).replace("\n", " ")
        return {"error": msg[-300:]}


def main():
    OUT["props"] = props()
    torch.npu.synchronize()

    n = 4096 * 4096
    cases = {}
    for dtype in (torch.float16, torch.float32):
        x = torch.randn(n, dtype=dtype, device="npu:0")
        tag = str(dtype).replace("torch.", "")
        rows = {}
        for block in (1024, 2048, 4096, 8192, 16384):
            rows[f"selu_b{block}"] = safe(lambda b=block: k_selu[(n // b,)](x, BLOCK=b))
        for block in (8192, 16384):
            rows[f"copy_b{block}"] = safe(lambda b=block: k_copy[(n // b,)](x, BLOCK=b))
        for block, g in ((8192, 20), (8192, 32), (8192, 40), (8192, 64), (4096, 32), (4096, 40)):
            ntile = n // block
            if ntile % g:
                rows[f"skip_gs_b{block}_g{g}"] = "not divisible"
                continue
            nt = ntile // g
            rows[f"selu_gs_b{block}_g{g}_nt{nt}"] = safe(
                lambda b=block, gg=g, nn=nt: k_selu_gs[(gg,)](x, nn, BLOCK=b))
            rows[f"copy_gs_b{block}_g{g}_nt{nt}"] = safe(
                lambda b=block, gg=g, nn=nt: k_copy_gs[(gg,)](x, nn, BLOCK=b))
        cases[tag] = rows
        del x
    OUT["big"] = cases

    small = {}
    for dtype in (torch.float16, torch.float32):
        x = torch.randn(1024, dtype=dtype, device="npu:0")
        tag = str(dtype).replace("torch.", "")
        small[tag] = {
            "triton_launch_b1024": safe(lambda: k_selu[(1,)](x, BLOCK=1024)),
            "triton_launch_b512": safe(lambda: k_selu[(2,)](x, BLOCK=512)),
        }
        del x
    OUT["small"] = small

    for mid in (65536, 1048576):
        x = torch.randn(mid, dtype=torch.float32, device="npu:0")
        row = {}
        for block in (1024, 8192):
            row[f"selu_b{block}"] = safe(lambda b=block: k_selu[(mid // b,)](x, BLOCK=b))
        row["selu_gs_b1024_g32"] = safe(
            lambda: k_selu_gs[(32,)](x, mid // 1024 // 32, BLOCK=1024))
        OUT[f"mid_{mid}"] = row
        del x

    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "selu_geom.json"), "w") as fh:
        json.dump(OUT, fh, indent=2, default=str)
    print(json.dumps(OUT, indent=2, default=str))


if __name__ == "__main__":
    main()
