"""Bounded target diagnostic: decompose the large-tensor cost of the fmod chain.

For 16.7M-element fp16/fp32 tensors, measure (a) a read/read/write roofline,
(b) divide-only, (c) the truncation chain without a divide, and (d) full
formulations with mask elision, tile-size sweep and two truncation spellings.
Every full formulation is checked elementwise against the NPU reference.
"""
import json
import os
import time

import torch
import torch_npu
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
RES = {}
DEV = torch.npu.current_device()
N = 16777216


@triton.jit
def k_add(A, B, N: tl.constexpr, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        a = tl.load(A + o).to(tl.float32)
        b = tl.load(B + o).to(tl.float32)
        tl.store(A + o, a + b)
    else:
        m = o < N
        a = tl.load(A + o, m, other=0).to(tl.float32)
        b = tl.load(B + o, m, other=1).to(tl.float32)
        tl.store(A + o, a + b, m)


@triton.jit
def k_div(A, B, N: tl.constexpr, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + o).to(tl.float32)
    b = tl.load(B + o).to(tl.float32)
    tl.store(A + o, a / b)


@triton.jit
def k_nodiv(A, B, N: tl.constexpr, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + o).to(tl.float32)
    b = tl.load(B + o).to(tl.float32)
    q = a * b
    magnitude = tl.floor(tl.abs(q))
    t = tl.where(q < 0, -magnitude, magnitude)
    tl.store(A + o, a - t * b)


@triton.jit
def k_full(A, B, N: tl.constexpr, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        a = tl.load(A + o).to(tl.float32)
        b = tl.load(B + o).to(tl.float32)
        q = a / b
        magnitude = tl.floor(tl.abs(q))
        t = tl.where(q < 0, -magnitude, magnitude)
        tl.store(A + o, a - t * b)
    else:
        m = o < N
        a = tl.load(A + o, m, other=0).to(tl.float32)
        b = tl.load(B + o, m, other=1).to(tl.float32)
        q = a / b
        magnitude = tl.floor(tl.abs(q))
        t = tl.where(q < 0, -magnitude, magnitude)
        tl.store(A + o, a - t * b, m)


@triton.jit
def k_seed(A, B, N: tl.constexpr, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + o).to(tl.float32)
    b = tl.load(B + o).to(tl.float32)
    q = a / b
    magnitude = tl.floor(tl.abs(q))
    sign = q.to(tl.int32, bitcast=True) & (-2147483648)
    sign = tl.where(magnitude == 0.0, 0, sign)
    t = (magnitude.to(tl.int32, bitcast=True) | sign).to(tl.float32, bitcast=True)
    tl.store(A + o, a - t * b)


@triton.jit
def k_trunc(A, B, N: tl.constexpr, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + o).to(tl.float32)
    b = tl.load(B + o).to(tl.float32)
    q = a / b
    t = tl.math.trunc(q) + 0.0
    tl.store(A + o, a - t * b)


def make(n, dtype, seed=1729):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, generator=g).to(device=DEV, dtype=dtype)
    B = (torch.randn(n, generator=g).abs() + 0.05).to(device=DEV, dtype=dtype)
    return A, B


def bench(fn, iters=40):
    for _ in range(10):
        fn()
    torch.npu.synchronize()
    s = torch.npu.Event(enable_timing=True)
    e = torch.npu.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.npu.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


TOL = {torch.float16: 1.5e-2, torch.bfloat16: 1.5e-2, torch.float32: 1e-4}

CONFIGS = [
    ("full_4096_mask", k_full, 4096, False),
    ("full_4096_even", k_full, 4096, True),
    ("full_2048_even", k_full, 2048, True),
    ("full_8192_even", k_full, 8192, True),
    ("full_16384_even", k_full, 16384, True),
    ("seed_4096_even", k_seed, 4096, True),
    ("trunc_4096_even", k_trunc, 4096, True),
]

for dt in (torch.float16, torch.float32):
    tag = str(dt).replace('torch.', '')
    A0, B = make(N, dt)
    R = A0.clone()
    R.fmod_(B)
    torch.npu.synchronize()
    Rf = R.float()
    tol = TOL[dt]
    out = {}

    def run_kernel(kern, block, even, A):
        grid = (N + block - 1) // block
        fn = lambda: kern[(grid,)](A, B, N, block, even, enable_fp_fusion=False)
        fn()
        torch.npu.synchronize()
        return fn

    for name, kern, block, even in CONFIGS:
        try:
            A = A0.clone()
            fn = run_kernel(kern, block, even, A)
            d = (A.float() - Rf).abs()
            thr = tol + tol * Rf.abs()
            nbad = int((d > thr).sum().item())
            nbit = int((A != R).sum().item())
            out[name] = {"us": bench(fn), "bad": nbad, "bit_diff": nbit}
        except Exception as exc:
            out[name] = {"error": f"{type(exc).__name__}: {str(exc)[:220]}"}
    for name, kern in (("roofline_add", k_add), ("div_only", k_div), ("chain_nodiv", k_nodiv)):
        try:
            A = A0.clone()
            out[name] = {"us": bench(run_kernel(kern, 4096, True, A))}
        except Exception as exc:
            out[name] = {"error": f"{type(exc).__name__}: {str(exc)[:220]}"}
    mb = N * A0.element_size() * 3 / 1e6
    out["bytes_MB"] = mb
    RES[tag] = out

with open(os.path.join(ART, "probe_dev.json"), "w") as fh:
    json.dump(RES, fh, indent=1)
print(json.dumps(RES, indent=1))
