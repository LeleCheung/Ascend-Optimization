"""Bounded target probe #2 for log2_: which term dominates the 4096x4096 cases?

Compares, at identical element counts, a pure copy kernel (memory floor), the
current float32-staged tl.log2, native-dtype tl.log2 (no staging), staged
tl.log * 1/ln2, and num_warps variants. Timed with the harness methodology
(median perf_counter over a sync-free loop of launches).
"""

import os
import time
import traceback

import torch
import triton
import triton.language as tl

DEV = "npu:0"
torch.npu.set_device(0)
ART = os.environ.get("KGS_DEBUG_ARTIFACTS") or "."
if ART and not os.path.isdir(ART):
    os.makedirs(ART, exist_ok=True)


@triton.jit
def _copy(x_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(x_ptr + offs, x)


@triton.jit
def _log2_staged(x_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.log2(x.to(tl.float32))
    tl.store(x_ptr + offs, y.to(x_ptr.dtype.element_ty))


@triton.jit
def _log2_native(x_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.log2(x)
    tl.store(x_ptr + offs, y)


@triton.jit
def _log_staged(x_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.log(x.to(tl.float32)) * 1.4426950408889634
    tl.store(x_ptr + offs, y.to(x_ptr.dtype.element_ty))


def bench(fn, iters, trials=5):
    for _ in range(min(iters, 20)):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(trials):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) / iters)
    ts.sort()
    return ts[len(ts) // 2] * 1e6


def case(name, x, kern, block, warps, iters):
    grid = (x.numel() // block,)
    try:
        kern[grid](x, BLOCK=block, num_warps=warps)
        torch.npu.synchronize()
    except Exception:
        print(f"{name:32s} ERROR: {traceback.format_exc()[-300:]}", flush=True)
        return None
    us = bench(lambda: kern[grid](x, BLOCK=block, num_warps=warps), iters)
    nbytes = x.numel() * x.element_size() * 2
    print(f"{name:32s} blk={block:6d} w={warps} {us:8.2f} us {nbytes/us/1e3:6.2f} GB/s",
          flush=True)
    return us


BIG = 4096 * 4096

xf = torch.rand(BIG, device=DEV, dtype=torch.float16) + 0.5
x3 = torch.rand(BIG, device=DEV, dtype=torch.float32) + 0.5
print("--- fp16 16.7M ---", flush=True)
case("copy(mem floor)", xf, _copy, 16384, 4, 20)
case("log2 native dtype", xf, _log2_native, 16384, 4, 20)
case("log2 fp32 staged", xf, _log2_staged, 16384, 4, 20)
case("log2 staged w=1", xf, _log2_staged, 16384, 1, 20)
case("log2 staged w=8", xf, _log2_staged, 16384, 8, 20)
case("log fp32 staged *1/ln2", xf, _log_staged, 16384, 4, 20)
case("log2 native blk=8192", xf, _log2_native, 8192, 4, 20)

print("--- fp32 16.7M ---", flush=True)
case("copy(mem floor)", x3, _copy, 8192, 4, 20)
case("log2 native(=fp32)", x3, _log2_native, 8192, 4, 20)
case("log2 fp32 staged", x3, _log2_staged, 8192, 4, 20)
case("log2 staged w=1", x3, _log2_staged, 8192, 1, 20)
case("log fp32 staged *1/ln2", x3, _log_staged, 8192, 4, 20)

print("--- 1M control ---", flush=True)
x1 = torch.rand(1024 * 1024, device=DEV, dtype=torch.float16) + 0.5
case("copy 1M fp16", x1, _copy, 16384, 4, 60)
case("log2 staged 1M fp16", x1, _log2_staged, 16384, 4, 60)
case("log2 native 1M fp16", x1, _log2_native, 16384, 4, 60)

print("--- torch reference calibration ---", flush=True)
print("torch.log2_ fp16 16.7M us =", bench(lambda: xf.log2_(), 20), flush=True)
print("torch.log2_ fp32 16.7M us =", bench(lambda: x3.log2_(), 20), flush=True)
print("done", flush=True)
