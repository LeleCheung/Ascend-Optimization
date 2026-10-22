"""Bounded target probe: where does the ~18.5us full-call time go for SELU in place,
and how expensive is tl.exp vs tl.exp2 on device for f16/bf16/f32."""
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
res = {"torch": torch.__version__, "triton": triton.__version__, "items": {}, "errors": []}
dev = "npu:0"
print("device", torch.npu.get_device_name(0), flush=True)


def bench(fn, iters=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def put(key, fn):
    try:
        res["items"][key] = fn()
    except Exception as exc:  # noqa
        res["items"][key] = None
        res["errors"].append({"key": key, "error": repr(exc), "tb": traceback.format_exc()[-1500:]})


# ---------------------------------------------------------------- candidate host path
sys.path.insert(0, "tmp")
import main as cand  # noqa: E402

CASES = [
    ("f16_1024", 1024, torch.float16),
    ("bf16_1M", 1048576, torch.bfloat16),
    ("f32_16M", 16777216, torch.float32),
    ("f16_16M", 16777216, torch.float16),
    ("bf16_16M", 16777216, torch.bfloat16),
]

for name, n, dt in CASES:
    A = torch.zeros(n, dtype=dt, device=dev)
    block = cand._pick_block(n, dt)
    grid = n // block
    masked = bool(n % block)
    key = (name, n, dt)
    try:
        cand.run(A)  # compile + populate cache
        torch.npu.synchronize()
    except Exception as exc:
        res["errors"].append({"key": name + ":warm", "error": repr(exc)})
        continue
    entry = cand._fast_get((n, A.data_ptr() & 15))
    if entry is None or entry[0] is not dt:
        res["errors"].append({"key": name + ":cache", "error": "fast path not populated"})
        continue
    launch = entry[1]
    ptr = A.data_ptr()
    # introspect the cached launch closure
    info = {}
    try:
        cells = dict(zip(launch.__code__.co_freevars, [c.cell_contents for c in (launch.__closure__ or ())]))
        call = cells.get("call")
        function = cells.get("function")
        metadata = cells.get("metadata")
        stream = cells.get("stream")
        gridv = cells.get("grid")
        addr_type = (launch.__kwdefaults__ or {}).get("_address")
        info["grid"] = gridv
        info["has_call"] = call is not None
        res["items"][name + ":closure"] = info
    except Exception as exc:
        res["errors"].append({"key": name + ":closure", "error": repr(exc)})
        continue
    addr = addr_type(ptr)
    try:
        res["items"][name + ":launcher_attrs"] = {
            "type_launch": type(call).__name__,
            "type_run": type(entry).__name__,
            "has_register": bool(getattr(entry, "enable_msprof_register_tensor", False)),
            "compile_only": bool(getattr(entry, "compile_only", False)),
        }
    except Exception:
        pass

    put(name + ":ref_torch_selu", lambda A=A: bench(lambda: torch.selu_(A)))
    put(name + ":full_run", lambda A=A: bench(lambda: cand.run(A)))
    put(name + ":std_launch", lambda A=A, g=grid, b=block: bench(
        lambda: cand._selu_nomask[g](A, BLOCK=b)))
    put(name + ":c_launch_only", lambda c=call, g=gridv, s=stream, f=function, m=metadata, a=addr: bench(
        lambda: c(g, 1, 1, s(), f, m, None, None, None, a, None)))
    sval = stream()
    put(name + ":c_launch_streamcached", lambda c=call, g=gridv, s=sval, f=function, m=metadata, a=addr: bench(
        lambda: c(g, 1, 1, s, f, m, None, None, None, a, None)))
    put(name + ":addr_construct", lambda at=addr_type, p=ptr: bench(lambda: at(p)))
    put(name + ":python_prefix", lambda A=A: bench(lambda: (
        A.numel(), A.data_ptr(), cand._fast_get((A.numel(), A.data_ptr() & 15)),
        A.is_contiguous())))
    put(name + ":python_prefix_full", lambda A=A: bench(lambda: (
        A.numel(), A.data_ptr(), cand._fast_get((A.numel(), A.data_ptr() & 15))[0],
        A.is_contiguous())))
    del A

# ---------------------------------------------------------------- device math variants
@triton.jit
def k_exp_f32(X, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * (tl.exp(x) - 1.0))
    tl.store(X + offs, y)


@triton.jit
def k_exp2_f32(X, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * (tl.exp2(x * 1.4426950408889634) - 1.0))
    tl.store(X + offs, y)


@triton.jit
def k_plain_f32(X, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * x)
    tl.store(X + offs, y)


@triton.jit
def k_exp_f16(X, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * (tl.exp(x) - 1.0))
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_exp2_f16(X, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * (tl.exp2(x * 1.4426950408889634) - 1.0))
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_plain_f16(X, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * x)
    tl.store(X + offs, y.to(X.dtype.element_ty))


# bf16 negative branch in fp16, positive branch in fp32 (no whole-tensor fp16 cast)
@triton.jit
def k_split_bf16(X, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    xf = x.to(tl.float32)
    xh = x.to(tl.float16)
    neg = (1.6732632423543772 * (tl.exp(xh) - 1.0)).to(tl.float32)
    y = 1.0507009873554805 * tl.where(xf > 0, xf, neg)
    tl.store(X + offs, y.to(tl.bfloat16))


MATH = {
    "f32_exp": (k_exp_f32, torch.float32, 8192),
    "f32_exp2": (k_exp2_f32, torch.float32, 8192),
    "f32_plain": (k_plain_f32, torch.float32, 8192),
    "f16_exp": (k_exp_f16, torch.float16, 16384),
    "f16_exp2": (k_exp2_f16, torch.float16, 16384),
    "f16_plain": (k_plain_f16, torch.float16, 16384),
    "bf16_split": (k_split_bf16, torch.bfloat16, 8192),
}
N = 16777216
tiny = {}
for nm, (kern, dt, block) in MATH.items():
    try:
        big = torch.zeros(N, dtype=dt, device=dev)
        sml = torch.zeros(block, dtype=dt, device=dev)
        kern[N // block, ](big, BLOCK=block)
        kern[1, ](sml, BLOCK=block)
        torch.npu.synchronize()
        t_big = bench(lambda: kern[N // block, ](big, BLOCK=block), iters=60)
        t_sml = bench(lambda: kern[1, ](sml, BLOCK=block), iters=60)
        res["items"][nm] = {"big_us": t_big, "tiny_us": t_sml, "device_us": t_big - t_sml}
        tiny[nm] = t_sml
        del big, sml
    except Exception as exc:
        res["items"][nm] = None
        res["errors"].append({"key": nm, "error": repr(exc), "tb": traceback.format_exc()[-1200:]})

# larger block feasibility on device for the f32/bf16 math path
for nm, (kern, dt, block) in (("f32_exp2_16384", (k_exp2_f32, torch.float32, 16384)),
                              ("bf16_split_16384", (k_split_bf16, torch.bfloat16, 16384))):
    try:
        big = torch.zeros(N, dtype=dt, device=dev)
        kern[N // block, ](big, BLOCK=block)
        torch.npu.synchronize()
        res["items"][nm] = {"big_us": bench(lambda: kern[N // block, ](big, BLOCK=block), iters=60)}
        del big
    except Exception as exc:
        res["items"][nm] = None
        res["errors"].append({"key": nm, "error": repr(exc)[:300]})

# ---------------------------------------------------------------- actual IR of the candidate kernel
try:
    A = torch.zeros(8192, dtype=torch.float32, device=dev)
    compiled = cand._selu_nomask.warmup(A, BLOCK=8192, grid=(1,))
    keys = list(getattr(compiled, "asm", {}).keys())
    res["items"]["asm_keys"] = keys
    for k in keys:
        txt = compiled.asm[k]
        if not isinstance(txt, str):
            continue
        low = txt.lower()
        hits = []
        for pat in ("vexp", "exp2", "elnexpm1", "expm1", "vln", "hivm.hir.exp", "math.exp", "llvm.exp"):
            c = low.count(pat)
            if c:
                hits.append((pat, c))
        res["items"]["asm_hits_" + str(k)] = hits
        res["items"]["asm_len_" + str(k)] = len(txt)
except Exception as exc:
    res["errors"].append({"key": "ir", "error": repr(exc), "tb": traceback.format_exc()[-1200:]})

with open(os.path.join(ART, "probe.json"), "w") as fh:
    json.dump(res, fh, indent=1, default=str)
print(json.dumps(res, indent=1, default=str)[:8000])
