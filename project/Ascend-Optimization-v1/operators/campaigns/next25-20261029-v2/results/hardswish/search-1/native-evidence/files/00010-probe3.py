"""Third bounded diagnosis: what sets the ~0.75 us/iteration fixed cost on
large tiles. Compares masked vs unmasked uniform-chunk loops, copy-only vs
math loops, native-dtype math, and reports full compile errors for big tiles.
"""
import json
import os
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = {"f16": {}, "f32": {}, "compile": {}}


def bench(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


@triton.jit
def k_mask(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for start in range(tl.program_id(0) * BLOCK, N, step):
        off = start + lane
        x = tl.load(X + off, off < N, other=0).to(tl.float32)
        inner = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
        tl.store(Y + off, x * inner * (1.0 / 6.0), off < N)


@triton.jit
def k_full(X, Y, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    base = tl.program_id(0) * BLOCK * ITERS
    lane = tl.arange(0, BLOCK)
    for i in range(ITERS):
        off = base + i * BLOCK + lane
        x = tl.load(X + off).to(tl.float32)
        inner = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
        tl.store(Y + off, x * inner * (1.0 / 6.0))


@triton.jit
def k_copy(X, Y, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    base = tl.program_id(0) * BLOCK * ITERS
    lane = tl.arange(0, BLOCK)
    for i in range(ITERS):
        off = base + i * BLOCK + lane
        tl.store(Y + off, tl.load(X + off))


@triton.jit
def k_native_full(X, Y, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    base = tl.program_id(0) * BLOCK * ITERS
    lane = tl.arange(0, BLOCK)
    for i in range(ITERS):
        off = base + i * BLOCK + lane
        x = tl.load(X + off)
        inner = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
        tl.store(Y + off, x * inner * (1.0 / 6.0))


KERNS = {"mask": k_mask, "full": k_full, "copy": k_copy, "native": k_native_full}


def run_variant(tag, bucket, kind, args_pos, grid, xx, yy, nb):
    kern = KERNS[kind]
    expected = xx if kind == "copy" else torch.nn.functional.hardswish(xx)
    try:
        kern.warmup(xx, yy, *args_pos, grid=(1,))
        yy.zero_()
        kern[(grid,)](xx, yy, *args_pos)
        torch.npu.synchronize()
        err = float((yy - expected).abs().max().item())
        t = bench(lambda: kern[(grid,)](xx, yy, *args_pos), 40, 15)
        OUT[bucket][tag] = {
            "us": t,
            "err": err,
            "logical_gbps": 2 * nb * xx.element_size() / t / 1e3,
        }
    except Exception as exc:
        OUT["compile"][f"{bucket}_{tag}"] = repr(exc)[:2500]


def main():
    nb = 4096 * 4096

    def sweep(bucket, dt, plan):
        xx = torch.randn(4096, 4096, device="npu", dtype=dt)
        yy = torch.empty_like(xx)
        OUT[bucket]["torch_ref_us"] = bench(
            lambda: torch.nn.functional.hardswish(xx), 40, 15)
        for kind, grid, blk in plan:
            iters = nb // (grid * blk)
            if iters * grid * blk != nb:
                continue
            if kind == "mask":
                run_variant(f"{kind}_g{grid}_b{blk}", bucket, kind, (nb, blk),
                            grid, xx, yy, nb)
            else:
                run_variant(f"{kind}_g{grid}_b{blk}_i{iters}", bucket, kind,
                            (blk, iters), grid, xx, yy, nb)

    sweep("f16", torch.float16, [
        ("mask", 40, 4096),
        ("full", 32, 4096), ("full", 64, 4096), ("full", 128, 4096),
        ("native", 64, 4096),
        ("full", 32, 8192), ("full", 32, 16384),
        ("copy", 64, 8192), ("copy", 32, 16384),
    ])
    sweep("f32", torch.float32, [
        ("mask", 40, 4096),
        ("full", 32, 4096), ("full", 64, 4096),
        ("full", 32, 8192),
    ])

    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "probe3.json"), "w") as fh:
        json.dump(OUT, fh, indent=2, sort_keys=True)
    print("WROTE probe3.json")


if __name__ == "__main__":
    main()
