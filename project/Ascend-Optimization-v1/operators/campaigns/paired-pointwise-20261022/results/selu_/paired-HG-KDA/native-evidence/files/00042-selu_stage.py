"""Final mechanism probe: pipelining depth, and formulations that fit bf16 in a
16384 tile.

Established mechanism: device time = (number of tiles) x ~1.5us / 40 vector
cores, independent of dtype and bytes.  A 16384 tile halves the tile count and
therefore halves the time (fp16 native measured 46.5us vs 82us at 8192), but the
UB budget (192KB, auto multi-buffered) blocks 16384 tiles for fp32 and for bf16
(bf16 misses by 32 bytes).

Probes:
  1. num_stages software pipelining -- can the 1.5us per-tile latency overlap?
     This is the only remaining lever for fp32 (stuck at an 8192 tile).
  2. bf16 formulations at a 16384 tile: exp2 form and scale-folded form.
  3. accuracy of every surviving candidate vs torch.selu.

Writes $KGS_DEBUG_ARTIFACTS/selu_stage.json
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


@triton.jit
def k_selu(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    xf = x.to(tl.float32)
    y = 1.0507009873554805 * tl.where(xf > 0, xf, 1.6732632423543772 * (tl.exp(xf) - 1.0))
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_selu_nat(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * (tl.exp(x) - 1.0))
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_selu_exp2(X, BLOCK: tl.constexpr):
    """exp2 form: avoids a separate fp32 exp staging buffer."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * (tl.exp2(x * 1.4426950408889634) - 1.0))
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_selu_fold(X, BLOCK: tl.constexpr):
    """Fold scale*alpha into one constant: one fewer multiply."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.where(x > 0, 1.0507009873554805 * x, 1.7580993408473766 * tl.exp(x) - 1.7580993408473766)
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_selu_pipe(X, NT, BLOCK: tl.constexpr, NS: tl.constexpr):
    pid = tl.program_id(0)
    g = tl.num_programs(0)
    for i in tl.range(0, NT, num_stages=NS):
        t = pid + i * g
        offs = t * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        xf = x.to(tl.float32)
        y = 1.0507009873554805 * tl.where(xf > 0, xf, 1.6732632423543772 * (tl.exp(xf) - 1.0))
        tl.store(X + offs, y.to(X.dtype.element_ty))


def bench(fn, iters=60, warmup=15):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    host_us = (time.perf_counter() - t0) / iters * 1e6
    torch.npu.synchronize()
    total_us = (time.perf_counter() - t0) / iters * 1e6
    return {"total_us": round(total_us, 2), "host_us": round(host_us, 2)}


def safe(fn):
    try:
        return bench(fn)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc).replace("\n", " ")[-140:]}


def accuracy(kern, dtype, block, label):
    n = 1 << 20
    src = torch.randn(n, dtype=dtype, device="npu:0")
    if dtype == torch.float16:
        src = src.clamp(-30.0, 30.0)
    ref = torch.selu(src.clone())
    got = src.clone()
    try:
        kern[(n // block,)](got, BLOCK=block)
        torch.npu.synchronize()
        d = (got.float() - ref.float()).abs()
        tol = 1e-4 if dtype == torch.float32 else 0.015
        ok = bool(torch.all(d <= tol + tol * ref.float().abs()))
        return {
            "label": label,
            "max_abs": round(d.max().item(), 7),
            "all_within_tol": ok,
            "bitwise_equal": bool(torch.equal(got, ref)),
        }
    except Exception as exc:  # noqa: BLE001
        return {"label": label, "error": str(exc).replace("\n", " ")[-140:]}


def main():
    n = 4096 * 4096

    # --- accuracy of the candidate formulations (1M elements each)
    OUT["accuracy"] = [
        accuracy(k_selu, torch.float16, 8192, "fp16 fp32math b8192"),
        accuracy(k_selu_nat, torch.float16, 16384, "fp16 native b16384"),
        accuracy(k_selu_exp2, torch.float16, 16384, "fp16 exp2 b16384"),
        accuracy(k_selu_fold, torch.float16, 16384, "fp16 fold b16384"),
        accuracy(k_selu, torch.bfloat16, 8192, "bf16 fp32math b8192"),
        accuracy(k_selu_exp2, torch.bfloat16, 16384, "bf16 exp2 b16384"),
        accuracy(k_selu_fold, torch.bfloat16, 16384, "bf16 fold b16384"),
        accuracy(k_selu, torch.float32, 8192, "fp32 fp32math b8192"),
    ]

    # --- fp32: pipelining depth is the only remaining lever
    x = torch.randn(n, dtype=torch.float32, device="npu:0")
    rows = {"selu_b8192": safe(lambda: k_selu[(n // 8192,)](x, BLOCK=8192))}
    for g, ns in ((64, 2), (64, 3), (64, 4), (128, 2), (128, 3), (256, 2)):
        ntile = n // 8192
        if ntile % g:
            continue
        rows[f"pipe_b8192_g{g}_ns{ns}"] = safe(
            lambda gg=g, nn=ntile // g, s=ns: k_selu_pipe[(gg,)](x, nn, BLOCK=8192, NS=s))
    OUT["fp32_pipe"] = rows
    del x

    # --- fp16 at the winning 16384 native tile, plus pipelined variants
    x = torch.randn(n, dtype=torch.float16, device="npu:0")
    rows = {
        "nat_b16384": safe(lambda: k_selu_nat[(n // 16384,)](x, BLOCK=16384)),
        "exp2_b16384": safe(lambda: k_selu_exp2[(n // 16384,)](x, BLOCK=16384)),
        "fold_b16384": safe(lambda: k_selu_fold[(n // 16384,)](x, BLOCK=16384)),
    }
    for g, ns in ((32, 2), (64, 2), (64, 3)):
        ntile = n // 16384
        if ntile % g:
            continue
        rows[f"pipe_b16384_g{g}_ns{ns}"] = safe(
            lambda gg=g, nn=ntile // g, s=ns: k_selu_pipe[(gg,)](x, nn, BLOCK=16384, NS=s))
    OUT["fp16_tile16384"] = rows
    del x

    # --- bf16: can any formulation fit a 16384 tile?
    x = torch.randn(n, dtype=torch.bfloat16, device="npu:0")
    OUT["bf16_tile16384"] = {
        "fp32math_b16384": safe(lambda: k_selu[(n // 16384,)](x, BLOCK=16384)),
        "nat_b16384": safe(lambda: k_selu_nat[(n // 16384,)](x, BLOCK=16384)),
        "exp2_b16384": safe(lambda: k_selu_exp2[(n // 16384,)](x, BLOCK=16384)),
        "fold_b16384": safe(lambda: k_selu_fold[(n // 16384,)](x, BLOCK=16384)),
        "fp32math_b8192_ref": safe(lambda: k_selu[(n // 8192,)](x, BLOCK=8192)),
    }
    del x

    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "selu_stage.json"), "w") as fh:
        json.dump(OUT, fh, indent=2, default=str)
    print(json.dumps(OUT, indent=2, default=str))


if __name__ == "__main__":
    main()
