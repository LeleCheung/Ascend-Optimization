"""Bounded target diagnostic: is the 16-bit element rate a codegen width limit?

Compares fp16/bf16 kernels that never widen to fp32 against the widened form,
plus tile-per-program and launch-argument variants.
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
def k_add_native(A, B, N: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(A + o, tl.load(A + o) + tl.load(B + o))


@triton.jit
def k_add_wide(A, B, N: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + o).to(tl.float32)
    b = tl.load(B + o).to(tl.float32)
    tl.store(A + o, a + b)


@triton.jit
def k_copy_native(A, B, N: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(A + o, tl.load(A + o))


@triton.jit
def k_add_native_gs(A, B, N: tl.constexpr, BLOCK: tl.constexpr, TILES: tl.constexpr):
    base = tl.program_id(0) * (BLOCK * TILES)
    for t in tl.static_range(TILES):
        o = base + t * BLOCK + tl.arange(0, BLOCK)
        tl.store(A + o, tl.load(A + o) + tl.load(B + o))


@triton.jit
def k_full_native(A, B, N: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + o)
    b = tl.load(B + o)
    q = a / b
    magnitude = tl.floor(tl.abs(q))
    t = tl.where(q < 0, -magnitude, magnitude)
    tl.store(A + o, a - t * b)


@triton.jit
def k_full_q16(A, B, N: tl.constexpr, BLOCK: tl.constexpr):
    """Div/trunc in 16-bit, product and subtraction in fp32."""
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + o)
    b = tl.load(B + o)
    q = a / b
    magnitude = tl.floor(tl.abs(q))
    t = tl.where(q < 0, -magnitude, magnitude)
    tl.store(A + o, a.to(tl.float32) - t.to(tl.float32) * b.to(tl.float32))


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


def time_loop(fn, iters=500):
    for _ in range(30):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


# ---- element-rate scaling on 16-bit storage ----
for dt in (torch.float16, torch.bfloat16):
    tag = str(dt).replace('torch.', '')
    A0, B = make(N, dt)
    row = {}
    for name, kern, block, extra in (
            ("add_native_4096", k_add_native, 4096, ()),
            ("add_native_8192", k_add_native, 8192, ()),
            ("add_wide_4096", k_add_wide, 4096, ()),
            ("copy_native_4096", k_copy_native, 4096, ()),
            ("add_native_gs2", k_add_native_gs, 4096, (2,)),
            ("add_native_gs4", k_add_native_gs, 4096, (4,)),
            ("full_native_4096", k_full_native, 4096, ()),
            ("full_q16_4096", k_full_q16, 4096, ())):
        try:
            A = A0.clone()
            grid = (N + block * (extra[0] if extra else 1) - 1) // (block * (extra[0] if extra else 1))
            fn = lambda: kern[(grid,)](A, B, N, block, *extra, enable_fp_fusion=False)
            fn()
            torch.npu.synchronize()
            row[name] = bench(fn)
        except Exception as exc:
            row[name] = f"{type(exc).__name__}: {str(exc)[:160]}"
    # correctness of the 16-bit-arithmetic formulations against the NPU reference
    R = A0.clone()
    R.fmod_(B)
    torch.npu.synchronize()
    for name, kern in (("full_native_4096", k_full_native), ("full_q16_4096", k_full_q16)):
        try:
            A = A0.clone()
            kern[(N // 4096,)](A, B, N, 4096, enable_fp_fusion=False)
            torch.npu.synchronize()
            d = (A.float() - R.float()).abs()
            thr = 1.5e-2 + 1.5e-2 * R.float().abs()
            row[name + "_bad"] = int((d > thr).sum().item())
            row[name + "_bitdiff"] = int((A != R).sum().item())
        except Exception as exc:
            row[name + "_bad"] = f"{type(exc).__name__}"
    RES[tag] = row

# ---- launch-argument sensitivity for the small-tensor host path ----
g = torch.Generator().manual_seed(1729)
a16 = torch.randn(1024, generator=g).to(device=DEV, dtype=torch.float16)
b16 = (torch.randn(1024, generator=g).abs() + 0.05).to(device=DEV, dtype=torch.float16)


@triton.jit
def k_flat(A, B, N: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    a = tl.load(A + o, m, other=0).to(tl.float32)
    b = tl.load(B + o, m, other=1).to(tl.float32)
    q = a / b
    mag = tl.floor(tl.abs(q))
    t = tl.where(q < 0, -mag, mag)
    tl.store(A + o, a - t * b, m)


comp = k_flat.warmup(a16, b16, 1024, 4096, grid=(1,), enable_fp_fusion=False)
launch = comp.run.launch
fn_c = comp.function
pmd = comp.packed_metadata
raw = torch_npu._C._npu_getCurrentRawStreamNoWait
pa, pb = a16.data_ptr(), b16.data_ptr()
RES["launch_full_args_us"] = time_loop(lambda: launch(1, 1, 1, raw(DEV), fn_c, pmd,
                                                      None, None, None, pa, pb, 1024, 4096))
RES["launch_fixed_stream_us"] = time_loop(lambda: launch(1, 1, 1, raw(DEV), fn_c, pmd,
                                                         None, None, None, pa, pb, 1024, 4096), 500)
for name, extra in (("launch_no_constexpr", (pa, pb)), ("launch_one_arg", (pa,))):
    try:
        launch(1, 1, 1, raw(DEV), fn_c, pmd, None, None, None, *extra)
        torch.npu.synchronize()
        RES[name + "_us"] = time_loop(lambda: launch(1, 1, 1, raw(DEV), fn_c, pmd,
                                                    None, None, None, *extra))
    except Exception as exc:
        RES[name + "_us"] = f"{type(exc).__name__}: {str(exc)[:160]}"

with open(os.path.join(ART, "probe_width.json"), "w") as fh:
    json.dump(RES, fh, indent=1)
print(json.dumps(RES, indent=1))
