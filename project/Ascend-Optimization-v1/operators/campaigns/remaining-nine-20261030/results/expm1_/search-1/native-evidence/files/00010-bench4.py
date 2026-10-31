"""Bounded target-side diagnostic #4 for expm1_ (direct compiled launcher).

Q1 host floor of the direct launch path
Q2 DMA floor (mul only) at 16KB vs 32KB tiles
Q3 does a buffer-trimmed (no signed-zero fix) formula fit a 32KB fp32 tile?
Q4 does deeper in-flight DMA (loop_unroll_factor / manual 4-way unroll) help?
"""
import json
import os
import time

import torch
import torch_npu
import triton
import triton.language as tl

ART_DIR = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
os.makedirs(ART_DIR, exist_ok=True)
OUT = os.path.join(ART_DIR, "bench4.json")
res = {"variants": []}


def save():
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)


@triton.jit
def _f_exp(x):
    xf = x.to(tl.float32)
    y = tl.exp(xf) - 1.0
    return tl.where(xf == 0.0, xf, y)


@triton.jit
def _f_exp_nofix(x):
    xf = x.to(tl.float32)
    return tl.exp(xf) - 1.0


@triton.jit
def _f_mul(x):
    return x.to(tl.float32) * 1.5


@triton.jit
def k_exp(a_ptr, n, BLOCK: tl.constexpr, NS: tl.constexpr):
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    full = n // BLOCK
    for t in tl.range(pid, full, G, num_stages=NS):
        off = t * BLOCK + lane
        x = tl.load(a_ptr + off)
        y = _f_exp(x)
        tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty))
    rem = n - full * BLOCK
    if rem > 0:
        if full % G == pid:
            off = full * BLOCK + lane
            m = off < n
            x = tl.load(a_ptr + off, mask=m, other=0.0)
            y = _f_exp(x)
            tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty), mask=m)


@triton.jit
def k_exp_unroll(a_ptr, n, BLOCK: tl.constexpr, NS: tl.constexpr, U: tl.constexpr):
    """tl.range loop_unroll_factor variant."""
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    full = n // BLOCK
    for t in tl.range(pid, full, G, num_stages=NS, loop_unroll_factor=U):
        off = t * BLOCK + lane
        x = tl.load(a_ptr + off)
        y = _f_exp(x)
        tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty))
    rem = n - full * BLOCK
    if rem > 0:
        if full % G == pid:
            off = full * BLOCK + lane
            m = off < n
            x = tl.load(a_ptr + off, mask=m, other=0.0)
            y = _f_exp(x)
            tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty), mask=m)


@triton.jit
def k_exp_nofix(a_ptr, n, BLOCK: tl.constexpr, NS: tl.constexpr):
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    full = n // BLOCK
    for t in tl.range(pid, full, G, num_stages=NS):
        off = t * BLOCK + lane
        x = tl.load(a_ptr + off)
        y = _f_exp_nofix(x)
        tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty))
    rem = n - full * BLOCK
    if rem > 0:
        if full % G == pid:
            off = full * BLOCK + lane
            m = off < n
            x = tl.load(a_ptr + off, mask=m, other=0.0)
            y = _f_exp_nofix(x)
            tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty), mask=m)


@triton.jit
def k_mul(a_ptr, n, BLOCK: tl.constexpr, NS: tl.constexpr):
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    full = n // BLOCK
    for t in tl.range(pid, full, G, num_stages=NS):
        off = t * BLOCK + lane
        x = tl.load(a_ptr + off)
        y = _f_mul(x)
        tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty))
    rem = n - full * BLOCK
    if rem > 0:
        if full % G == pid:
            off = full * BLOCK + lane
            m = off < n
            x = tl.load(a_ptr + off, mask=m, other=0.0)
            y = _f_mul(x)
            tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty), mask=m)


@triton.jit
def k_exp_u4(a_ptr, n, BLOCK: tl.constexpr, NS: tl.constexpr):
    """Manual 4-way unroll: four independent tiles in flight per iteration."""
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    full = n // BLOCK
    step = 4 * G
    stop = full - 3 * G
    for t in tl.range(pid, stop, step, num_stages=NS):
        o0 = t * BLOCK + lane
        o1 = o0 + G * BLOCK
        o2 = o1 + G * BLOCK
        o3 = o2 + G * BLOCK
        x0 = tl.load(a_ptr + o0)
        x1 = tl.load(a_ptr + o1)
        x2 = tl.load(a_ptr + o2)
        x3 = tl.load(a_ptr + o3)
        tl.store(a_ptr + o0, _f_exp(x0).to(a_ptr.dtype.element_ty))
        tl.store(a_ptr + o1, _f_exp(x1).to(a_ptr.dtype.element_ty))
        tl.store(a_ptr + o2, _f_exp(x2).to(a_ptr.dtype.element_ty))
        tl.store(a_ptr + o3, _f_exp(x3).to(a_ptr.dtype.element_ty))
    m_iters = tl.maximum((stop - pid + step - 1) // step, 0)
    t0 = pid + m_iters * step
    for t in tl.range(t0, full, G):
        off = t * BLOCK + lane
        x = tl.load(a_ptr + off)
        y = _f_exp(x)
        tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty))
    rem = n - full * BLOCK
    if rem > 0:
        if full % G == pid:
            off = full * BLOCK + lane
            m = off < n
            x = tl.load(a_ptr + off, mask=m, other=0.0)
            y = _f_exp(x)
            tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty), mask=m)


def _stream():
    raw = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
    if raw is not None:
        try:
            return raw(0)
        except TypeError:
            return raw()
    return torch.npu.current_stream(0).npu_stream


def mklaunch(kernel, A, n, BLOCK, NS, grid, U=None):
    kw = {"BLOCK": BLOCK, "NS": NS}
    if U is not None:
        kw["U"] = U
    ck = kernel.warmup(A, n, grid=(1,), **kw)
    launcher = ck.run
    debug = (getattr(launcher, "compile_only", False) or
             getattr(launcher, "enable_msprof_register_tensor", False) or
             getattr(launcher.metadata, "debug_enabled", False) or
             getattr(launcher.metadata, "debug", False))
    fn = launcher if debug else launcher.launch
    fk, pm = ck.function, ck.packed_metadata
    ptr = A.data_ptr()
    if U is None:
        def call():
            fn(int(grid), 1, 1, _stream(), fk, pm, None, None, None, ptr, n, BLOCK, NS)
    else:
        def call():
            fn(int(grid), 1, 1, _stream(), fk, pm, None, None, None, ptr, n, BLOCK, NS, U)
    return call


def bench(fn, iters=300, warmup=30):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters * 1e3)
    return best


def add(name, kernel, A, n, BLOCK, NS, grid, U=None):
    try:
        call = mklaunch(kernel, A, n, BLOCK, NS, grid, U)
        res["variants"].append({"name": name, "wall_ms": bench(call)})
    except Exception as e:  # noqa: BLE001
        msg = repr(e)
        res["variants"].append({"name": name, "error": msg[:200], "error_tail": msg[-260:]})
    save()


G = 40
tiny = torch.randn(1024, dtype=torch.float32, device="npu")
add("host_floor_fp32_1024", k_exp, tiny, 1024, 1024, 2, 1)
del tiny

n32 = 4096 * 4096
A32 = torch.randn(n32, dtype=torch.float32, device="npu")
add("fp32|exp_B4096_ns2_g40", k_exp, A32, n32, 4096, 2, G)
add("fp32|mul_B4096_ns2_g40", k_mul, A32, n32, 4096, 2, G)
add("fp32|mul_B8192_ns2_g40", k_mul, A32, n32, 8192, 2, G)
add("fp32|exp_nofix_B8192_ns2_g40", k_exp_nofix, A32, n32, 8192, 2, G)
add("fp32|expunroll_B4096_ns2_u4_g40", k_exp_unroll, A32, n32, 4096, 2, G, 4)
add("fp32|exp_u4_B4096_ns2_g40", k_exp_u4, A32, n32, 4096, 2, G)
add("fp32|exp_u4_B4096_ns1_g40", k_exp_u4, A32, n32, 4096, 1, G)
add("fp32|exp_u4_B2048_ns2_g40", k_exp_u4, A32, n32, 2048, 2, G)
del A32
save()

n16 = 4096 * 4096
A16 = torch.randn(n16, dtype=torch.float32).to(torch.float16).npu()
add("fp16|exp_B8192_ns2_g40", k_exp, A16, n16, 8192, 2, G)
add("fp16|exp_u4_B8192_ns2_g40", k_exp_u4, A16, n16, 8192, 2, G)
add("fp16|exp_u4_B4096_ns2_g40", k_exp_u4, A16, n16, 4096, 2, G)
del A16
save()

print(json.dumps(res, indent=1)[:7000])
