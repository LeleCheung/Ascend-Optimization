"""Probe 4: persistent grid-stride SELU on the 16M cases (per-program cost hypothesis),
plus a persistent copy floor."""
import json
import os
import sys
import time
import traceback

import torch
import torch_npu
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
res = {"items": {}, "errors": []}
dev = "npu:0"


def flush():
    try:
        with open(os.path.join(ART, "probe4.json"), "w") as fh:
            json.dump(res, fh, indent=1, default=str)
    except Exception:
        pass


def bench(fn, iters=80, warmup=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return round((time.perf_counter() - t0) / iters * 1e6, 3)


def put(key, fn):
    try:
        res["items"][key] = fn()
    except Exception as exc:  # noqa
        res["items"][key] = None
        res["errors"].append({"key": key, "error": repr(exc)[:400], "tb": traceback.format_exc()[-400:]})
    flush()


sys.path.insert(0, "tmp")
import main as cand  # noqa: E402

try:
    import torch_npu._C as _C
    res["items"]["soc"] = {
        "vector_core": int(_C._npu_getDeviceInfo(0, "vector_core_num")) if hasattr(_C, "_npu_getDeviceInfo") else None,
    }
except Exception as exc:
    res["items"]["soc"] = {"error": repr(exc)[:200]}
flush()


@triton.jit
def p_f32(X, TILES: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    for t in range(pid, TILES, G):
        offs = t * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        y = 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * (tl.exp(x) - 1.0))
        tl.store(X + offs, y)


@triton.jit
def p_f16(X, TILES: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    for t in range(pid, TILES, G):
        offs = t * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        y = 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * (tl.exp(x) - 1.0))
        tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def p_bf16(X, TILES: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    for t in range(pid, TILES, G):
        offs = t * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        xf = x.to(tl.float32)
        y = 1.0507009873554805 * tl.where(xf > 0, xf, 1.6732632423543772 * (tl.exp(xf) - 1.0))
        tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def p_copy(X, TILES: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    for t in range(pid, TILES, G):
        offs = t * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        tl.store(X + offs, x)


N = 16777216
PLAN = [
    ("f32_p_G20", p_f32, torch.float32, 8192, 20),
    ("f32_p_G40", p_f32, torch.float32, 8192, 40),
    ("f32_p_G80", p_f32, torch.float32, 8192, 80),
    ("copy_f32_p_G20", p_copy, torch.float32, 8192, 20),
    ("f16_p_G20", p_f16, torch.float16, 16384, 20),
    ("f16_p_G40", p_f16, torch.float16, 16384, 40),
    ("copy_f16_p_G20", p_copy, torch.float16, 16384, 20),
    ("bf16_p_G20", p_bf16, torch.bfloat16, 8192, 20),
    ("bf16_p_G40", p_bf16, torch.bfloat16, 8192, 40),
]
for nm, kern, dt, block, g in PLAN:
    try:
        A = torch.zeros(N, dtype=dt, device=dev)
        tiles = N // block
        compiled = kern.warmup(A, TILES=tiles, G=g, BLOCK=block, grid=(1,))
        call, fn, md = compiled.run.launch, compiled.function, compiled.packed_metadata
        sget = cand._resolve_stream(A.device.index)
        p = A.data_ptr()
        t = bench(lambda: call(g, 1, 1, sget(), fn, md, None, None, None, p, None, None, None))
        res["items"][nm] = {"us": t, "tiles": tiles, "grid": g, "block": block}
        del A
    except Exception as exc:
        res["items"][nm] = None
        res["errors"].append({"key": nm, "error": repr(exc)[:300]})
    flush()

# numeric check of the persistent kernel
try:
    A = torch.randn(8192 * 40 * 3, device=dev, dtype=torch.float32)
    ref = A.clone()
    torch.selu_(ref)
    tiles = A.numel() // 8192
    p_f32[(20,)](A, tiles, 20, BLOCK=8192)
    torch.npu.synchronize()
    res["items"]["persist_correctness"] = {
        "max_abs": float((A - ref).abs().max()),
        "n": int(A.numel()), "tiles": tiles,
    }
except Exception as exc:
    res["errors"].append({"key": "persist_correct", "error": repr(exc)[:300]})
flush()

print(json.dumps(res, indent=1, default=str)[:8000])
