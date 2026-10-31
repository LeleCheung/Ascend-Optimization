"""Bounded target probe #3 for log2_: persistent looped kernel vs single-tile.

Question: on 16.7M elements, does a persistent grid-stride looped kernel (many
tiles per program, pipelined DMA) recover the memory-path throughput that the
native op reaches, and does tl.log+scale stay at the copy floor there?
"""

import os
import time
import traceback

import torch
import triton
import triton.language as tl

DEV = "npu:0"
torch.npu.set_device(0)

INV_LN2 = 1.4426950408889634


@triton.jit
def _loop_copy(x_ptr, n, TILE: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    ntiles = n // TILE
    for t in tl.range(pid, ntiles, nprog, num_stages=2):
        offs = t * TILE + tl.arange(0, TILE)
        x = tl.load(x_ptr + offs)
        tl.store(x_ptr + offs, x)


@triton.jit
def _loop_log(x_ptr, n, TILE: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    ntiles = n // TILE
    for t in tl.range(pid, ntiles, nprog, num_stages=2):
        offs = t * TILE + tl.arange(0, TILE)
        x = tl.load(x_ptr + offs)
        y = tl.log(x.to(tl.float32)) * 1.4426950408889634
        tl.store(x_ptr + offs, y.to(x_ptr.dtype.element_ty))


@triton.jit
def _loop_log2(x_ptr, n, TILE: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    ntiles = n // TILE
    for t in tl.range(pid, ntiles, nprog, num_stages=2):
        offs = t * TILE + tl.arange(0, TILE)
        x = tl.load(x_ptr + offs)
        y = tl.log2(x.to(tl.float32))
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


def case(name, x, kern, tile, grid, iters=20):
    n = x.numel()
    try:
        kern[(grid,)](x, n, TILE=tile)
        torch.npu.synchronize()
    except Exception:
        print(f"{name:34s} ERROR {traceback.format_exc()[-260:]}", flush=True)
        return None
    us = bench(lambda: kern[(grid,)](x, n, TILE=tile), iters)
    nbytes = n * x.element_size() * 2
    print(f"{name:34s} tile={tile:6d} g={grid:4d} {us:8.2f} us {nbytes/us/1e3:6.2f} GB/s",
          flush=True)
    return us


BIG = 4096 * 4096
xf = torch.rand(BIG, device=DEV, dtype=torch.float16) + 0.5
x3 = torch.rand(BIG, device=DEV, dtype=torch.float32) + 0.5

print("--- fp16 16.7M looped ---", flush=True)
case("loop copy", xf, _loop_copy, 16384, 64)
case("loop log", xf, _loop_log, 16384, 64)
case("loop log", xf, _loop_log, 16384, 128)
case("loop log", xf, _loop_log, 16384, 20)
case("loop log", xf, _loop_log, 8192, 128)
case("loop log2", xf, _loop_log2, 16384, 64)

print("--- fp32 16.7M looped ---", flush=True)
case("loop copy", x3, _loop_copy, 16384, 64)
case("loop log", x3, _loop_log, 16384, 64)
case("loop log", x3, _loop_log, 16384, 128)
case("loop log", x3, _loop_log, 8192, 64)

print("--- 1M fp16 looped control ---", flush=True)
x1 = torch.rand(1024 * 1024, device=DEV, dtype=torch.float16) + 0.5
case("loop log 1M", x1, _loop_log, 16384, 64, 60)
case("loop log 1M", x1, _loop_log, 4096, 64, 60)

print("--- references ---", flush=True)
print("torch.log2_ fp16 16.7M us =", bench(lambda: xf.log2_(), 20), flush=True)
print("torch.log2_ fp32 16.7M us =", bench(lambda: x3.log2_(), 20), flush=True)
print("done", flush=True)
