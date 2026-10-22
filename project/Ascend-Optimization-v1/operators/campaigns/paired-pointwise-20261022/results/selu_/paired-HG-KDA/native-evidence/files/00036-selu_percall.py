"""Per-call latency probe (sync per call) for in-place SELU variants.

Motivation: the pipelined loop used before cannot resolve device vs host cost.
This measures full per-call latency the way a walltime harness does
(sync -> t0 -> call -> sync), and also cross-checks against a batched loop.

Variants probe the two remaining levers:
  * tile size (8192 vs 16384)  -> UB budget doubles with the fp32 math tile
  * programs per element (1 tile vs 2 tiles per program at the same tile size)
  * native low-precision math for fp16/bf16 (drops the fp32 tile)

Writes $KGS_DEBUG_ARTIFACTS/selu_percall.json
"""

import json
import os
import statistics
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
OUT = {}

SCALE = 1.0507009873554805
ALPHA = 1.6732632423543772


@triton.jit
def k_selu(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    xf = x.to(tl.float32)
    y = SCALE * tl.where(xf > 0, xf, ALPHA * (tl.exp(xf) - 1.0))
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_selu_native(X, BLOCK: tl.constexpr):
    """No fp32 tile: halves the UB footprint so BLOCK can double."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = SCALE * tl.where(x > 0, x, ALPHA * (tl.exp(x) - 1.0))
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_selu_x2(X, BLOCK: tl.constexpr):
    """Two sequential tiles per program: half the programs, same tile/UB size."""
    pid = tl.program_id(0)
    base = pid * (2 * BLOCK)
    for i in tl.static_range(2):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        xf = x.to(tl.float32)
        y = SCALE * tl.where(xf > 0, xf, ALPHA * (tl.exp(xf) - 1.0))
        tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_copy(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(X + offs, tl.load(X + offs))


def bench_sync(fn, iters=60, warmup=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return {"med_us": round(statistics.median(ts), 2), "min_us": round(ts[0], 2)}


def safe(fn, iters=60):
    try:
        return bench_sync(fn, iters=iters)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc).replace("\n", " ")[-220:]}


def main():
    n = 4096 * 4096

    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        tag = str(dtype).replace("torch.", "")
        ref = torch.randn(n, dtype=dtype, device="npu:0")
        rows = {}
        rows["torch_selu_inplace"] = safe(lambda: torch.selu_(ref))
        rows["torch_selu_outofplace"] = safe(lambda: torch.selu(ref))

        x = ref.clone()
        del ref
        torch.npu.synchronize()

        rows["selu_b8192"] = safe(lambda: k_selu[(n // 8192,)](x, BLOCK=8192))
        rows["selu_x2_b8192"] = safe(lambda: k_selu_x2[(n // 16384,)](x, BLOCK=8192))
        rows["selu_native_b8192"] = safe(lambda: k_selu_native[(n // 8192,)](x, BLOCK=8192))
        rows["selu_native_b16384"] = safe(lambda: k_selu_native[(n // 16384,)](x, BLOCK=16384))
        rows["copy_b8192"] = safe(lambda: k_copy[(n // 8192,)](x, BLOCK=8192))
        rows["copy_b16384"] = safe(lambda: k_copy[(n // 16384,)](x, BLOCK=16384))

        # correctness cross-check of the native path against torch.selu on fresh data
        probe = torch.randn(65536, dtype=dtype, device="npu:0")
        expect = torch.selu(probe.clone())
        got = probe.clone()
        try:
            k_selu_native[(65536 // 1024,)](got, BLOCK=1024)
            torch.npu.synchronize()
            err = (got.float() - expect.float()).abs().max().item()
            rows["native_maxabs_err_65536"] = round(err, 6)
        except Exception as exc:  # noqa: BLE001
            rows["native_maxabs_err_65536"] = repr(exc)[-160:]
        del probe, expect, got, x
    OUT[f"big_{tag}"] = rows

    # small / mid shapes, fp32, per-call latency
    for nmid in (1024, 65536, 1048576):
        x = torch.randn(nmid, dtype=torch.float32, device="npu:0")
        row = {"torch_selu_inplace": safe(lambda: torch.selu_(x))}
        for block in (1024, 8192):
            if nmid % block == 0:
                row[f"selu_b{block}"] = safe(lambda b=block: k_selu[(nmid // b,)](x, BLOCK=b))
        row["copy_b1024"] = safe(lambda: k_copy[(nmid // 1024,)](x, BLOCK=1024))
        OUT[f"mid_{nmid}"] = row
        del x

    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "selu_percall.json"), "w") as fh:
        json.dump(OUT, fh, indent=2, default=str)
    print(json.dumps(OUT, indent=2, default=str))


if __name__ == "__main__":
    main()
