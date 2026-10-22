"""Probe 3: does an in-program multi-tile loop reduce the per-wave overhead on the big
cases? Also: plain-int address launch, and what the C launcher actually is."""
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
        with open(os.path.join(ART, "probe3.json"), "w") as fh:
            json.dump(res, fh, indent=1, default=str)
    except Exception:
        pass


def bench(fn, iters=100, warmup=10):
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
        res["errors"].append({"key": key, "error": repr(exc)[:400], "tb": traceback.format_exc()[-500:]})
    flush()


sys.path.insert(0, "tmp")
import main as cand  # noqa: E402


def direct_parts(kern, A, block, kw=None):
    kw = kw or {}
    compiled = kern.warmup(A, BLOCK=block, grid=(1,), **kw)
    return compiled.run.launch, compiled.function, compiled.packed_metadata, compiled


def time_direct(kern, A, block, iters=80, extra=None, kw=None):
    """extra: list of additional runtime args after the address (e.g. tail count)."""
    grid = A.numel() // block
    call, fn, md, compiled = direct_parts(kern, A, block, kw)
    sget = cand._resolve_stream(A.device.index)
    at = cand._hgkda_address_type(A.numel())
    p = A.data_ptr()
    args = extra or []

    def go():
        call(grid, 1, 1, sget(), fn, md, None, None, None, at(p), *args, None)

    return bench(go, iters=iters), compiled


# ------------------------------------------------------------------ launcher introspection
try:
    A = torch.zeros(8192, dtype=torch.float32, device=dev)
    call, fn, md, compiled = direct_parts(cand._selu_nomask, A, 8192)
    info = {
        "call_type": type(call).__name__,
        "call_module": getattr(call, "__module__", None),
        "call_qualname": getattr(call, "__qualname__", None),
        "run_type": type(compiled.run).__name__,
        "run_dir": [k for k in dir(compiled.run) if not k.startswith("__")][:40],
        "md_type": type(md).__name__,
        "md_len": len(md) if hasattr(md, "__len__") else None,
        "md_sample": str(md)[:400],
        "fn_type": type(fn).__name__,
    }
    try:
        import inspect
        info["call_src"] = inspect.getsource(call)[:1500]
    except Exception as exc:
        info["call_src"] = "unavailable: " + repr(exc)[:120]
    try:
        info["run_src"] = inspect.getsource(type(compiled.run))[:1200]
    except Exception as exc:
        info["run_src"] = "unavailable: " + repr(exc)[:120]
    res["items"]["launcher"] = info
except Exception as exc:
    res["errors"].append({"key": "launcher", "error": repr(exc), "tb": traceback.format_exc()[-600:]})
flush()

# ------------------------------------------------------------------ plain-int address
try:
    A = torch.randn(1024, device=dev, dtype=torch.float16)
    ref = A.clone()
    torch.selu_(ref)
    cand.run(A)
    torch.npu.synchronize()
    ok_wrap = torch.allclose(A, ref, atol=1e-2, rtol=1e-2)
    B = torch.randn(1024, device=dev, dtype=torch.float16)
    refB = B.clone()
    torch.selu_(refB)
    block = cand._pick_block(1024, torch.float16)
    call, f, md, compiled = direct_parts(cand._selu_nomask, B, block)
    sget = cand._resolve_stream(B.device.index)
    p = B.data_ptr()
    plain_ok = None
    try:
        call(1, 1, 1, sget(), f, md, None, None, None, p, None)
        torch.npu.synchronize()
        plain_ok = bool(torch.allclose(B, refB, atol=1e-2, rtol=1e-2))
    except Exception as exc:
        plain_ok = "error: " + repr(exc)[:200]
    res["items"]["plain_int_address"] = {
        "wrapped_ok": bool(ok_wrap),
        "plain_int_result_ok": plain_ok,
        "t_plain_int": bench(lambda: call(1, 1, 1, sget(), f, md, None, None, None, p, None)),
        "t_wrapped": bench(lambda: call(1, 1, 1, sget(), f, md, None, None, None, cand._hgkda_address_type(1024)(p), None)),
        "t_plain_int_cached_stream": None,
    }
    sval = sget()
    res["items"]["plain_int_address"]["t_plain_int_cached_stream"] = bench(
        lambda: call(1, 1, 1, sval, f, md, None, None, None, p, None))
except Exception as exc:
    res["errors"].append({"key": "plain_int", "error": repr(exc), "tb": traceback.format_exc()[-600:]})
flush()

# ------------------------------------------------------------------ multi-tile loop kernels
@triton.jit
def k_f32_loop(X, TILES, BLOCK: tl.constexpr):
    base = tl.program_id(0) * (BLOCK * TILES)
    for t in range(0, TILES):
        offs = base + t * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        y = 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * (tl.exp(x) - 1.0))
        tl.store(X + offs, y)


@triton.jit
def k_bf16_loop(X, TILES, BLOCK: tl.constexpr):
    base = tl.program_id(0) * (BLOCK * TILES)
    for t in range(0, TILES):
        offs = base + t * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        xf = x.to(tl.float32)
        y = 1.0507009873554805 * tl.where(xf > 0, xf, 1.6732632423543772 * (tl.exp(xf) - 1.0))
        tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_f16_loop(X, TILES, BLOCK: tl.constexpr):
    base = tl.program_id(0) * (BLOCK * TILES)
    for t in range(0, TILES):
        offs = base + t * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        y = 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * (tl.exp(x) - 1.0))
        tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_copy_loop(X, TILES, BLOCK: tl.constexpr):
    base = tl.program_id(0) * (BLOCK * TILES)
    for t in range(0, TILES):
        offs = base + t * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        tl.store(X + offs, x)


N = 16777216
CASES = [
    ("f32_loop2_8192", k_f32_loop, torch.float32, 8192, 2),
    ("f32_loop4_8192", k_f32_loop, torch.float32, 8192, 4),
    ("copy_loop2_8192_f32", k_copy_loop, torch.float32, 8192, 2),
    ("copy_loop4_16384_f16", k_copy_loop, torch.float16, 16384, 4),
    ("f16_loop2_16384", k_f16_loop, torch.float16, 16384, 2),
    ("f16_loop4_16384", k_f16_loop, torch.float16, 16384, 4),
    ("bf16_loop2_8192", k_bf16_loop, torch.bfloat16, 8192, 2),
    ("bf16_loop4_8192", k_bf16_loop, torch.bfloat16, 8192, 4),
]
for nm, kern, dt, block, tiles in CASES:
    try:
        A = torch.zeros(N, dtype=dt, device=dev)
        grid = N // (block * tiles)
        call, fn, md, compiled = direct_parts(kern, A, block)
        sget = cand._resolve_stream(A.device.index)
        at = cand._hgkda_address_type(N)
        p = A.data_ptr()
        t = bench(lambda: call(grid, 1, 1, sget(), fn, md, None, None, None, at(p), tiles, None), iters=80)
        res["items"][nm] = {"us": t, "grid": grid}
        del A
    except Exception as exc:
        res["items"][nm] = None
        res["errors"].append({"key": nm, "error": repr(exc)[:300]})
    flush()

# numeric check of the loop kernel shape (small random tensor, tiles=2)
try:
    A = torch.randn(8192 * 2 * 4, device=dev, dtype=torch.float32)
    ref = A.clone()
    torch.selu_(ref)
    k_f32_loop[4, ](A, 2, BLOCK=8192)
    torch.npu.synchronize()
    res["items"]["loop_correctness"] = {
        "max_abs": float((A - ref).abs().max()),
        "max_rel": float(((A - ref).abs() / ref.abs().clamp_min(1e-6)).max()),
    }
except Exception as exc:
    res["errors"].append({"key": "loop_correct", "error": repr(exc)[:300]})
flush()

print(json.dumps(res, indent=1, default=str)[:9000])
