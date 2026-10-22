"""Bounded target probe: where does the ~18.5us SELU full call go, and how expensive
is tl.exp vs tl.exp2 on device for f16/bf16/f32."""
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


def flush():
    try:
        with open(os.path.join(ART, "probe.json"), "w") as fh:
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
        res["errors"].append({"key": key, "error": repr(exc), "tb": traceback.format_exc()[-800:]})
    flush()


print("device", torch.npu.get_device_name(0), flush=True)
sys.path.insert(0, "tmp")
import main as cand  # noqa: E402

from triton.runtime import driver  # noqa: E402

CASES = [
    ("f16_1024", 1024, torch.float16),
    ("bf16_1M", 1048576, torch.bfloat16),
    ("f16_16M", 16777216, torch.float16),
    ("f32_16M", 16777216, torch.float32),
    ("bf16_16M", 16777216, torch.bfloat16),
]

for name, n, dt in CASES:
    try:
        A = torch.zeros(n, dtype=dt, device=dev)
        cand.run(A)  # compile + populate cache through the real entry point
        torch.npu.synchronize()
        put(name + ":ref_torch_selu", lambda A=A: bench(lambda: torch.selu_(A)))
        put(name + ":full_run", lambda A=A: bench(lambda: cand.run(A)))
        block = cand._pick_block(n, dt)
        grid = n // block
        put(name + ":std_launch", lambda A=A, g=grid, b=block: bench(
            lambda: cand._selu_nomask[g](A, BLOCK=b)))
        # reconstruct the direct-launch pieces exactly as _make_launch does
        with torch.no_grad():
            compiled = cand._selu_nomask.warmup(A, BLOCK=block, grid=(1,))
        call = compiled.run.launch
        function = compiled.function
        metadata = compiled.packed_metadata
        stream_fn = driver.active.get_current_stream(A.device.index)
        addr_type = cand._hgkda_address_type(n)
        ptr = A.data_ptr()
        addr = addr_type(ptr)
        sval = stream_fn()
        res["items"][name + ":meta"] = {"block": block, "grid": grid,
                                        "launch_type": type(call).__name__,
                                        "stream_type": type(sval).__name__,
                                        "sval": int(sval) if isinstance(sval, int) else str(sval)}
        flush()
        put(name + ":c_launch_prebuilt_addr", lambda c=call, g=grid, f=function, m=metadata, a=addr: bench(
            lambda: c(g, 1, 1, sval, f, m, None, None, None, a, None)))
        put(name + ":c_launch_addr_percall", lambda c=call, g=grid, f=function, m=metadata, at=addr_type, s=sval, p=ptr: bench(
            lambda: c(g, 1, 1, s, f, m, None, None, None, at(p), None)))
        put(name + ":stream_get", lambda sf=stream_fn: bench(lambda: sf()))
        put(name + ":addr_construct", lambda at=addr_type, p=ptr: bench(lambda: at(p)))
        put(name + ":python_prefix", lambda A=A: bench(lambda: (
            A.numel(), A.data_ptr(), cand._fast_get((A.numel(), A.data_ptr() & 15)),
            A.is_contiguous())))
        # host cost when the launch itself is a no-op: measures pure wrapper overhead
        put(name + ":python_only_run_equiv", lambda A=A: bench(lambda: (
            A.numel(), A.data_ptr(), A.dtype, A.is_contiguous())))
        del A
    except Exception as exc:
        res["errors"].append({"key": name, "error": repr(exc), "tb": traceback.format_exc()[-800:]})
        flush()

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


@triton.jit
def k_split_bf16(X, BLOCK: tl.constexpr):
    """negative branch exp in fp16, positive branch kept in fp32 (no whole-tensor cast)"""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    xf = x.to(tl.float32)
    xh = x.to(tl.float16)
    neg = (1.6732632423543772 * (tl.exp(xh) - 1.0)).to(tl.float32)
    y = 1.0507009873554805 * tl.where(xf > 0, xf, neg)
    tl.store(X + offs, y.to(tl.bfloat16))


@triton.jit
def k_split_bf16_exp2(X, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    xf = x.to(tl.float32)
    xh = x.to(tl.float16)
    neg = (1.6732632423543772 * (tl.exp2(xh * 1.4426950408889634) - 1.0)).to(tl.float32)
    y = 1.0507009873554805 * tl.where(xf > 0, xf, neg)
    tl.store(X + offs, y.to(tl.bfloat16))


N = 16777216
MATH = [
    ("f32_exp", k_exp_f32, torch.float32, 8192),
    ("f32_exp2", k_exp2_f32, torch.float32, 8192),
    ("f32_plain", k_plain_f32, torch.float32, 8192),
    ("f16_exp", k_exp_f16, torch.float16, 16384),
    ("f16_exp2", k_exp2_f16, torch.float16, 16384),
    ("f16_plain", k_plain_f16, torch.float16, 16384),
    ("bf16_split_exp", k_split_bf16, torch.bfloat16, 8192),
    ("bf16_split_exp2", k_split_bf16_exp2, torch.bfloat16, 8192),
]
for nm, kern, dt, block in MATH:
    try:
        big = torch.zeros(N, dtype=dt, device=dev)
        sml = torch.zeros(block, dtype=dt, device=dev)
        kern[N // block, ](big, BLOCK=block)
        kern[1, ](sml, BLOCK=block)
        torch.npu.synchronize()
        t_big = bench(lambda: kern[N // block, ](big, BLOCK=block), iters=60)
        t_sml = bench(lambda: kern[1, ](sml, BLOCK=block), iters=60)
        res["items"][nm] = {"big_us": t_big, "tiny_us": t_sml, "device_us": round(t_big - t_sml, 3)}
        del big, sml
    except Exception as exc:
        res["items"][nm] = None
        res["errors"].append({"key": nm, "error": repr(exc)[:400], "tb": traceback.format_exc()[-600:]})
    flush()

for nm, kern, dt, block in (("f32_exp2_16384", k_exp2_f32, torch.float32, 16384),
                            ("bf16_split_exp2_16384", k_split_bf16_exp2, torch.bfloat16, 16384),
                            ("f16_exp2_32768", k_exp2_f16, torch.float16, 32768)):
    try:
        big = torch.zeros(N, dtype=dt, device=dev)
        kern[N // block, ](big, BLOCK=block)
        torch.npu.synchronize()
        res["items"][nm] = {"big_us": bench(lambda: kern[N // block, ](big, BLOCK=block), iters=60)}
        del big
    except Exception as exc:
        res["items"][nm] = None
        res["errors"].append({"key": nm, "error": repr(exc)[:400]})
    flush()

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
        for pat in ("vexp", "exp2", "expm1", "vln", "hivm.hir.exp", "math.exp", "llvm.exp", "vexp2"):
            c = low.count(pat)
            if c:
                hits.append([pat, c])
        res["items"]["asm_hits_" + str(k)] = hits
        res["items"]["asm_len_" + str(k)] = len(txt)
except Exception as exc:
    res["errors"].append({"key": "ir", "error": repr(exc), "tb": traceback.format_exc()[-800:]})
flush()
print(json.dumps(res, indent=1, default=str)[:9000])
