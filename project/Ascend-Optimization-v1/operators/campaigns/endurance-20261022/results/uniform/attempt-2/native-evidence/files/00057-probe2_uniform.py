"""Probe 2: is there a fixed per-program cost on this backend, and is tl.philox the driver?

Compares: aclnn fill_ (control), store-only kernel, scale-only kernel, tl.philox kernel,
and a manual uint32 Philox4x32-10 kernel, sweeping grid and BLOCK.
"""
import json
import os
import pathlib
import traceback

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from flag_gems.utils.random_utils import uint_to_uniform_float

ART = pathlib.Path(os.environ.get("KGS_DEBUG_ARTIFACTS", "."))
ART.mkdir(parents=True, exist_ok=True)
R = {}

W0NEG = -1640531527  # 0x9E3779B9 as int32
W1NEG = -1150833019  # 0xBB67AE85 as int32
M0L = 0x1F53
M0H = 0xD251
M1L = 0x8D57
M1H = 0xCD9E


@triton.jit
def _mul16(a, B0: tl.constexpr, B1: tl.constexpr):
    a0 = a & 0xFFFF
    a1 = a >> 16
    u0 = a0 * B0
    t = a1 * B0 + (u0 >> 16)
    w1 = a0 * B1 + (t & 0xFFFF)
    hi = a1 * B1 + (t >> 16) + (w1 >> 16)
    lo = u0 ^ (w1 << 16)
    return hi, lo


@triton.jit
def _philox16(seed, c0, c1):
    k0 = seed.to(tl.uint32)
    k1 = (seed >> 32).to(tl.uint32)
    c2 = c0 * 0
    c3 = c2
    for _ in tl.static_range(10):
        hi0, lo0 = _mul16(c0, M0L, M0H)
        hi1, lo1 = _mul16(c2, M1L, M1H)
        c0 = hi1 ^ c1 ^ k0
        c1 = lo1
        c2 = hi0 ^ c3 ^ k1
        c3 = lo0
        k0 = k0 + tl.full((), W0NEG, tl.int32).to(tl.uint32)
        k1 = k1 + tl.full((), W1NEG, tl.int32).to(tl.uint32)
    return c0, c1, c2, c3


@triton.jit
def _next_below(x, DT: tl.constexpr):
    zero = x == 0
    if DT == tl.float32:
        bits = x.to(tl.int32, bitcast=True)
        step = tl.where(x >= 0, -1, 1).to(tl.int32)
        stepped = (bits + step).to(tl.float32, bitcast=True)
        tiny = tl.full((), 1, tl.int32).to(tl.float32, bitcast=True)
        return tl.where(zero, -tiny, stepped).to(tl.float32)
    else:
        bits = x.to(tl.int16, bitcast=True)
        step = tl.where(x >= 0, -1, 1).to(tl.int16)
        stepped = (bits + step).to(DT, bitcast=True)
        tiny = tl.full((), 1, tl.int16).to(DT, bitcast=True)
        return tl.where(zero, -tiny, stepped).to(DT)


@triton.jit
def _emit(r, out_ptr, off, N, from_, to):
    OUT_DT: tl.constexpr = out_ptr.dtype.element_ty
    to_dt = (r * 0.0 + to).to(OUT_DT)
    from_dt = (r * 0.0 + from_).to(OUT_DT)
    hi = _next_below(to_dt, OUT_DT)
    hi = tl.where(to_dt > from_dt, hi, to_dt)
    v = tl.minimum(r.to(OUT_DT), hi)
    tl.store(out_ptr + off, v, mask=off < N)


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def k_const(out_ptr, N, philox_seed, philox_offset, from_, to, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK * 4 + tl.arange(0, BLOCK)
    zero = (philox_offset * 0 + pid * 0).to(tl.float32)
    v = (zero + 0.5).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + off, v, mask=off < N)
    tl.store(out_ptr + off + BLOCK, v, mask=off + BLOCK < N)
    tl.store(out_ptr + off + BLOCK * 2, v, mask=off + BLOCK * 2 < N)
    tl.store(out_ptr + off + BLOCK * 3, v, mask=off + BLOCK * 3 < N)


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def k_scale(out_ptr, N, philox_seed, philox_offset, from_, to, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK * 4 + tl.arange(0, BLOCK)
    a = tl.arange(0, BLOCK).to(tl.uint32)
    base = (a << 16) ^ a ^ (pid * 0 + 1)
    rg = uint_to_uniform_float(base) * (to - from_) + from_
    rg2 = uint_to_uniform_float(base + 1) * (to - from_) + from_
    rg3 = uint_to_uniform_float(base + 2) * (to - from_) + from_
    rg4 = uint_to_uniform_float(base + 3) * (to - from_) + from_
    _emit(rg, out_ptr, off, N, from_, to)
    _emit(rg2, out_ptr, off + BLOCK, N, from_, to)
    _emit(rg3, out_ptr, off + BLOCK * 2, N, from_, to)
    _emit(rg4, out_ptr, off + BLOCK * 3, N, from_, to)


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def k_tlphilox(out_ptr, N, philox_seed, philox_offset, from_, to, BLOCK: tl.constexpr):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    off = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    _emit(uint_to_uniform_float(r0) * (to - from_) + from_, out_ptr, off, N, from_, to)
    _emit(uint_to_uniform_float(r1) * (to - from_) + from_, out_ptr, off + BLOCK, N, from_, to)
    _emit(uint_to_uniform_float(r2) * (to - from_) + from_, out_ptr, off + BLOCK * 2, N, from_, to)
    _emit(uint_to_uniform_float(r3) * (to - from_) + from_, out_ptr, off + BLOCK * 3, N, from_, to)


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def k_manual16(out_ptr, N, philox_seed, philox_offset, from_, to, BLOCK: tl.constexpr):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    r0, r1, r2, r3 = _philox16(philox_seed, c0, c1)
    off = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    _emit(uint_to_uniform_float(r0) * (to - from_) + from_, out_ptr, off, N, from_, to)
    _emit(uint_to_uniform_float(r1) * (to - from_) + from_, out_ptr, off + BLOCK, N, from_, to)
    _emit(uint_to_uniform_float(r2) * (to - from_) + from_, out_ptr, off + BLOCK * 2, N, from_, to)
    _emit(uint_to_uniform_float(r3) * (to - from_) + from_, out_ptr, off + BLOCK * 3, N, from_, to)


def bench_us(fn, iters):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    e0 = torch.npu.Event(enable_timing=True)
    e1 = torch.npu.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.npu.synchronize()
    return e0.elapsed_time(e1) * 1e3 / iters


def kern_case(name, kernel, N, BLOCK, dtype=torch.float32, iters=50):
    try:
        out = torch.empty(N, dtype=dtype, device="npu:0")
        grid = (triton.cdiv(N, BLOCK * 4),)
        fn = lambda: kernel[grid](out, N, 1234, 0, 0.0, 1.0, BLOCK=BLOCK)
        R[f"{name}_N{N}_B{BLOCK}_g{grid[0]}"] = round(bench_us(fn, iters), 2)
    except Exception as exc:
        R[f"{name}_N{N}_B{BLOCK}"] = "ERROR: " + type(exc).__name__ + ": " + str(exc)[:600]


def control(name, fn, iters=50):
    try:
        R[name] = round(bench_us(fn, iters), 2)
    except Exception as exc:
        R[name] = "ERROR: " + type(exc).__name__ + ": " + str(exc)[:300]


# aclnn controls: allocation and a simple in-place elementwise op
def ctl_fill(n):
    x = torch.empty(n, dtype=torch.float32, device="npu:0")
    control(f"ctl_fill_{n}", lambda: x.fill_(0.5))


ctl_fill(4096)
ctl_fill(1048576)
control("ctl_empty_4096", lambda: torch.empty(4096, dtype=torch.float32, device="npu:0"))

# grid=1 sweeps: fixed cost vs BLOCK
kern_case("const", k_const, 1024, 256)
kern_case("const", k_const, 4096, 1024)
kern_case("scale", k_scale, 4096, 1024)
kern_case("tlphilox", k_tlphilox, 2048, 512)
kern_case("tlphilox", k_tlphilox, 4096, 1024)
kern_case("manual16", k_manual16, 4096, 1024)

# multi-program: does cost scale with waves (per-program cost) or with bytes?
kern_case("const", k_const, 64000, 1024)          # grid 16  (1 wave)
kern_case("const", k_const, 1048576, 1024, iters=20)   # grid 256 (7 waves)
kern_case("manual16", k_manual16, 1048576, 1024, iters=20)


def bitcheck():
    N = 4096
    out_a = torch.empty(N, dtype=torch.float32, device="npu:0")
    out_b = torch.empty(N, dtype=torch.float32, device="npu:0")
    grid = (triton.cdiv(N, 1024 * 4),)
    k_tlphilox[grid](out_a, N, 987654321, 4096, 0.0, 1.0, BLOCK=1024)
    k_manual16[grid](out_b, N, 987654321, 4096, 0.0, 1.0, BLOCK=1024)
    torch.npu.synchronize()
    return {
        "manual16_equal": bool(torch.equal(out_a, out_b)),
        "manual16_maxdiff": float((out_a - out_b).abs().max()),
        "range": [float(out_a.min()), float(out_a.max())],
        "mean": float(out_a.double().mean()),
    }


try:
    R["bitcheck"] = bitcheck()
except Exception as exc:
    R["bitcheck"] = "ERROR: " + type(exc).__name__ + ": " + str(exc)[:800] + "\n" + traceback.format_exc()[-1500:]

(ART / "probe2-uniform.json").write_text(json.dumps(R, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(R, indent=1, ensure_ascii=False)[:8000])
