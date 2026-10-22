"""Probe 2: decouple MTE transfer size from compute-tile size for in-place SELU.

Evidence so far (n=16777216, device us): copy b16384 40-42, copy b32768 22-28,
fp16 native selu b16384 46.8, bf16 fp32-math b8192 84.8, fp32 fp32-math b8192 98.4.
Device time tracks the NUMBER of MTE transfers; UB capacity caps BLOCK at 16384
(2-byte) / 8192 (4-byte) because the fp32 math needs three full-width tiles live.

This probe tests whether a single big load/store combined with a per-half compute
(tl.split / tl.join) lowers the transfer count without exceeding UB, and whether
bf16 can borrow fp16 math width.
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

_C1 = 1.0507009873554805
_C2 = 1.6732632423543772


@triton.jit
def k_base(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    xf = x.to(tl.float32)
    y = 1.0507009873554805 * tl.where(xf > 0, xf, 1.6732632423543772 * (tl.exp(xf) - 1.0))
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_nat(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * (tl.exp(x) - 1.0))
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_halves(X, BLOCK: tl.constexpr):
    """One BLOCK-wide load/store; math on two BLOCK/2 halves (fp32 domain)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    x2 = tl.reshape(x, (BLOCK // 2, 2))
    a, b = tl.split(x2)
    af = a.to(tl.float32)
    bf = b.to(tl.float32)
    ya = 1.0507009873554805 * tl.where(af > 0, af, 1.6732632423543772 * (tl.exp(af) - 1.0))
    yb = 1.0507009873554805 * tl.where(bf > 0, bf, 1.6732632423543772 * (tl.exp(bf) - 1.0))
    y = tl.reshape(tl.join(ya.to(X.dtype.element_ty), yb.to(X.dtype.element_ty)), (BLOCK,))
    tl.store(X + offs, y)


@triton.jit
def k_via_f16(X, BLOCK: tl.constexpr):
    """bf16 input, math done at fp16 width (exact cast in, rounding at the end)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs).to(tl.float16)
    y = 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * (tl.exp(x) - 1.0))
    tl.store(X + offs, y.to(X.dtype.element_ty))


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


def build(kern, x, block, n, opts):
    grid = n // block
    compiled = kern.warmup(x, BLOCK=block, grid=(1,), **opts)
    launcher = compiled.run
    call = launcher.launch
    fn_h = compiled.function
    meta = compiled.packed_metadata
    get = stream_getter()

    def invoke(arg):
        call(grid, 1, 1, get(), fn_h, meta, None, None, None, arg, None)

    return invoke


def main():
    n = 4096 * 4096
    OUT["n"] = n
    tol = {"float16": 0.015, "bfloat16": 0.015, "float32": 0.0001}

    plan = {
        "float16": [
            ("nat_b16384", k_nat, 16384, {}),
            ("halves_b32768", k_halves, 32768, {}),
        ],
        "bfloat16": [
            ("base_b8192", k_base, 8192, {}),
            ("halves_b16384", k_halves, 16384, {}),
            ("halves_b32768", k_halves, 32768, {}),
            ("via_f16_b16384", k_via_f16, 16384, {}),
            ("via_f16_b32768", k_via_f16, 32768, {}),
        ],
        "float32": [
            ("base_b8192", k_base, 8192, {}),
            ("halves_b16384", k_halves, 16384, {}),
            ("halves_b32768", k_halves, 32768, {}),
        ],
    }

    for tag, variants in plan.items():
        dtype = getattr(torch, tag)
        rows = {}
        base = torch.randn(n, dtype=dtype, device="npu:0")
        ref = torch.nn.functional.selu(base.clone().float())
        for name, kern, block, opts in variants:
            row = {"block": block}
            try:
                invoke = build(kern, base, block, n, opts)
                x = base.clone()
                invoke(x)
                torch.npu.synchronize()
                d = (x.float() - ref).abs()
                thr = tol[tag] + tol[tag] * ref.abs()
                row["max_abs_err"] = round(d.max().item(), 8)
                row["bad_elems"] = int((d > thr).sum().item())
                row["n_elems"] = n
                del x
                y = base.clone()
                row.update(bench(lambda: invoke(y)))
                del y
            except Exception as exc:  # noqa: BLE001
                row["error"] = str(exc).replace("\n", " ")[-240:]
            rows[name] = row
            print(tag, name, row, flush=True)
        OUT[tag] = rows
        del base, ref

    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "split_probe.json"), "w") as fh:
        json.dump(OUT, fh, indent=2, default=str)
    print(json.dumps(OUT, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
