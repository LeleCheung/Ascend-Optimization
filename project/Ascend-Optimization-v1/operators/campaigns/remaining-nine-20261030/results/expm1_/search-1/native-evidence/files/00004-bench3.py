"""Bounded target-side diagnostic #3 for expm1_.

Measures through the *direct compiled launcher* (the path the candidate uses) so
the ~50us Triton JIT host binder does not mask device time.

Questions:
  Q1 what is the largest UB-fitting tile per dtype (buffer-count probe)?
  Q2 does a bigger DMA tile or deeper pipelining beat B4096/ns2?
  Q3 what is the pure-DMA floor vs the exp math cost?
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
OUT = os.path.join(ART_DIR, "bench3.json")
res = {"variants": [], "notes": []}


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


def _stream():
    raw = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
    if raw is not None:
        try:
            return raw(0)
        except TypeError:
            return raw()
    return torch.npu.current_stream(0).npu_stream


def mklaunch(kernel, A, n, BLOCK, NS, grid):
    ck = kernel.warmup(A, n, BLOCK=BLOCK, NS=NS, grid=(1,))
    launcher = ck.run
    debug = (getattr(launcher, "compile_only", False) or
             getattr(launcher, "enable_msprof_register_tensor", False) or
             getattr(launcher.metadata, "debug_enabled", False) or
             getattr(launcher.metadata, "debug", False))
    fn = launcher if debug else launcher.launch
    fk, pm = ck.function, ck.packed_metadata
    ptr = A.data_ptr()
    dev = A.device.index

    def call():
        fn((grid,), 1, 1, _stream(), fk, pm, None, None, None, ptr, n, BLOCK, NS)

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


def add(name, kernel, A, n, BLOCK, NS, grid):
    try:
        call = mklaunch(kernel, A, n, BLOCK, NS, grid)
        res["variants"].append({"name": name, "wall_ms": bench(call)})
    except Exception as e:  # noqa: BLE001
        res["variants"].append({"name": name, "error": repr(e)[:220]})
    save()


G = 40

# ---- Q3 host cost floor: tiny tensor through the direct launcher
tiny = torch.randn(1024, dtype=torch.float32, device="npu")
add("host_floor_fp32_1024", k_exp, tiny, 1024, 1024, 2, 1)
del tiny

# ---- fp32 4096x4096
n32 = 4096 * 4096
A32 = torch.randn(n32, dtype=torch.float32, device="npu")
add("fp32|exp_B4096_ns2_g40", k_exp, A32, n32, 4096, 2, G)
add("fp32|mul_B4096_ns2_g40", k_mul, A32, n32, 4096, 2, G)
add("fp32|exp_B8192_ns2_g40", k_exp, A32, n32, 8192, 2, G)
add("fp32|exp_nofix_B8192_ns2_g40", k_exp_nofix, A32, n32, 8192, 2, G)
add("fp32|mul_B8192_ns2_g40", k_mul, A32, n32, 8192, 2, G)
add("fp32|exp_B2048_ns3_g40", k_exp, A32, n32, 2048, 3, G)
add("fp32|exp_B2048_ns4_g40", k_exp, A32, n32, 2048, 4, G)
del A32
save()

# ---- fp16 4096x4096
n16 = 4096 * 4096
A16 = torch.randn(n16, dtype=torch.float32).to(torch.float16).npu()
add("fp16|exp_B8192_ns2_g40", k_exp, A16, n16, 8192, 2, G)
add("fp16|exp_B16384_ns2_g40", k_exp, A16, n16, 16384, 2, G)
add("fp16|mul_B16384_ns2_g40", k_mul, A16, n16, 16384, 2, G)
add("fp16|exp_B8192_ns3_g40", k_exp, A16, n16, 8192, 3, G)
del A16
save()

# ---- 1M fp32 through the direct launcher (medium case)
n1 = 1 << 20
A1 = torch.randn(n1, dtype=torch.float32, device="npu")
add("fp32_1M|exp_B4096_ns2_g40", k_exp, A1, n1, 4096, 2, G)
del A1
save()

print(json.dumps(res, indent=1)[:7000])
