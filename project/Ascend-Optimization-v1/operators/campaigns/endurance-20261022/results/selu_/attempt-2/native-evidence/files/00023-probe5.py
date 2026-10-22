"""Probe 5: which option lets the persistent SELU loop fit UB, and how fast is it."""
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
        with open(os.path.join(ART, "probe5.json"), "w") as fh:
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


sys.path.insert(0, "tmp")
import main as cand  # noqa: E402


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


N = 16777216
PLAN = [
    ("f32_G20_b8192_s1", p_f32, torch.float32, 8192, 20, {"num_stages": 1}),
    ("f32_G20_b8192_s2", p_f32, torch.float32, 8192, 20, {"num_stages": 2}),
    ("f32_G20_b8192_mbF", p_f32, torch.float32, 8192, 20, {"multibuffer": False}),
    ("f32_G40_b8192_s1", p_f32, torch.float32, 8192, 40, {"num_stages": 1}),
    ("f32_G20_b4096_s1", p_f32, torch.float32, 4096, 20, {"num_stages": 1}),
    ("f16_G20_b16384_s1", p_f16, torch.float16, 16384, 20, {"num_stages": 1}),
    ("f16_G40_b16384_s1", p_f16, torch.float16, 16384, 40, {"num_stages": 1}),
    ("bf16_G20_b8192_s1", p_bf16, torch.bfloat16, 8192, 20, {"num_stages": 1}),
    ("bf16_G20_b16384_s1", p_bf16, torch.bfloat16, 16384, 20, {"num_stages": 1}),
    ("bf16_G40_b16384_s1", p_bf16, torch.bfloat16, 16384, 40, {"num_stages": 1}),
]
for nm, kern, dt, block, g, kw in PLAN:
    try:
        A = torch.zeros(N, dtype=dt, device=dev)
        tiles = N // block
        compiled = kern.warmup(A, TILES=tiles, G=g, BLOCK=block, grid=(1,), **kw)
        call, fn, md = compiled.run.launch, compiled.function, compiled.packed_metadata
        sget = cand._resolve_stream(A.device.index)
        p = A.data_ptr()
        t = bench(lambda: call(g, 1, 1, sget(), fn, md, None, None, None, p, None, None, None))
        res["items"][nm] = {"us": t, "tiles": tiles, "grid": g, "block": block, "opt": kw}
        del A
    except Exception as exc:
        msg = repr(exc)
        res["items"][nm] = None
        res["errors"].append({"key": nm, "opt": kw,
                              "ub": "ub overflow" in msg,
                              "error": msg[-700:]})
    flush()

print(json.dumps(res, indent=1, default=str)[:9000])
