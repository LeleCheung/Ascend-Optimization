"""Probe 2: clean device-time comparisons through the direct launcher (13us host floor)
plus host-overhead decomposition with the candidate's own stream accessor."""
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
        with open(os.path.join(ART, "probe2.json"), "w") as fh:
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


def direct_parts(kern, A, block):
    compiled = kern.warmup(A, BLOCK=block, grid=(1,))
    return compiled.run.launch, compiled.function, compiled.packed_metadata


def time_direct(kern, A, block, iters=100):
    grid = A.numel() // block
    call, fn, md = direct_parts(kern, A, block)
    sget = cand._resolve_stream(A.device.index)
    at = cand._hgkda_address_type(A.numel())
    p = A.data_ptr()

    def go():
        call(grid, 1, 1, sget(), fn, md, None, None, None, at(p), None)

    return bench(go, iters=iters)


# ------------------------------------------------------------------ host decomposition
for name, n, dt in (("f16_1024", 1024, torch.float16), ("f32_1024", 1024, torch.float32)):
    try:
        A = torch.zeros(n, dtype=dt, device=dev)
        cand.run(A)
        torch.npu.synchronize()
        block = cand._pick_block(n, dt)
        grid = n // block
        call, fn, md = direct_parts(cand._selu_nomask, A, block)
        sget = cand._resolve_stream(A.device.index)
        at = cand._hgkda_address_type(n)
        p = A.data_ptr()
        addr = at(p)
        sval = sget()
        h = {}
        h["ref_torch_selu"] = bench(lambda: torch.selu_(A))
        h["full_run"] = bench(lambda: cand.run(A))
        h["closure_launch"] = bench(lambda: call(grid, 1, 1, sget(), fn, md, None, None, None, at(p), None))
        h["c_launch_stream_cached"] = bench(lambda: call(grid, 1, 1, sval, fn, md, None, None, None, addr, None))
        h["std_entry"] = bench(lambda: cand._selu_nomask[(grid,)](A, BLOCK=block))
        h["stream_get"] = bench(lambda: sget())
        h["addr_construct"] = bench(lambda: at(p))
        h["py_prefix_numel_ptr"] = bench(lambda: (A.numel(), A.data_ptr()))
        h["py_prefix_contig"] = bench(lambda: A.is_contiguous())
        h["py_prefix_dict"] = bench(lambda: (cand._fast_get((n, p & 15)), A.dtype, A.is_contiguous()))
        h["sval_type"] = str(type(sval).__name__)
        res["items"][name] = h
        del A
    except Exception as exc:
        res["errors"].append({"key": name, "error": repr(exc), "tb": traceback.format_exc()[-700:]})
    flush()

# ------------------------------------------------------------------ kernels
@triton.jit
def k_copy(X, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    tl.store(X + offs, x)


@triton.jit
def k_f32(X, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * (tl.exp(x) - 1.0))
    tl.store(X + offs, y)


@triton.jit
def k_f32_fma(X, BLOCK: tl.constexpr):
    """one fewer vector op: c1*exp(x)+c0 folded, positive branch scaled once"""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    e = tl.exp(x)
    y = tl.where(x > 0, 1.0507009873554805 * x, 1.7580993408473766 * e - 1.7580993408473766)
    tl.store(X + offs, y)


@triton.jit
def k_f16(X, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * (tl.exp(x) - 1.0))
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_bf16_up(X, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    xf = x.to(tl.float32)
    y = 1.0507009873554805 * tl.where(xf > 0, xf, 1.6732632423543772 * (tl.exp(xf) - 1.0))
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_bf16_split(X, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    xf = x.to(tl.float32)
    xh = x.to(tl.float16)
    neg = (1.6732632423543772 * (tl.exp(xh) - 1.0)).to(tl.float32)
    y = 1.0507009873554805 * tl.where(xf > 0, xf, neg)
    tl.store(X + offs, y.to(tl.bfloat16))


@triton.jit
def k_bf16_f16(X, BLOCK: tl.constexpr):
    """negatives and the alpha*(exp-1) product fully in fp16, positives in fp32"""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    xf = x.to(tl.float32)
    xh = x.to(tl.float16)
    neg = (1.0507009873554805 * 1.6732632423543772) * (tl.exp(xh) - 1.0)
    y = tl.where(xf > 0, 1.0507009873554805 * xf, neg.to(tl.float32))
    tl.store(X + offs, y.to(tl.bfloat16))


N = 16777216
PLAN = [
    ("copy_f32", k_copy, torch.float32, 8192),
    ("f32_8192", k_f32, torch.float32, 8192),
    ("f32_4096", k_f32, torch.float32, 4096),
    ("f32_2048", k_f32, torch.float32, 2048),
    ("f32fma_8192", k_f32_fma, torch.float32, 8192),
    ("copy_f16", k_copy, torch.float16, 16384),
    ("f16_16384", k_f16, torch.float16, 16384),
    ("f16_8192", k_f16, torch.float16, 8192),
    ("copy_bf16", k_copy, torch.bfloat16, 8192),
    ("bf16up_8192", k_bf16_up, torch.bfloat16, 8192),
    ("bf16split_8192", k_bf16_split, torch.bfloat16, 8192),
    ("bf16f16_8192", k_bf16_f16, torch.bfloat16, 8192),
    ("bf16f16_16384", k_bf16_f16, torch.bfloat16, 16384),
    ("bf16f16_4096", k_bf16_f16, torch.bfloat16, 4096),
]

for nm, kern, dt, block in PLAN:
    try:
        A = torch.zeros(N, dtype=dt, device=dev)
        t = time_direct(kern, A, block, iters=80)
        res["items"][nm] = {"us": t}
        del A
    except Exception as exc:
        res["items"][nm] = None
        res["errors"].append({"key": nm, "error": repr(exc)[:300]})
    flush()

# small-tensor floor through the same direct path (isolates host overhead)
for nm, kern, dt, block in (("floor_f32", k_f32, torch.float32, 8192),
                            ("floor_bf16", k_bf16_f16, torch.bfloat16, 8192)):
    try:
        A = torch.zeros(block, dtype=dt, device=dev)
        res["items"][nm] = {"us": time_direct(kern, A, block, iters=100)}
        del A
    except Exception as exc:
        res["items"][nm] = None
        res["errors"].append({"key": nm, "error": repr(exc)[:300]})
    flush()

print(json.dumps(res, indent=1, default=str)[:9000])
