"""Bounded perf probe for cross_out on Ascend NPU.

Distinguishes, for shape (1048576, 3):
  * host+launch overhead               -> k_null
  * aligned elementwise ceiling        -> k_copy
  * current 8 shifted-load kernel      -> k_flat8
  * 8 loads but all 32B-aligned        -> k_flat8aligned  (wrong math, perf only)
  * minimal-traffic stride-3 kernel    -> k_gather3       (6 loads / 3 stores)
"""
import json
import os
import time

import torch
import triton
import triton.language as tl

DEV = "npu"
NROW = 1048576
N = 3 * NROW


@triton.jit
def k_null(p):
    tl.store(p + tl.program_id(0), 0.0)


@triton.jit
def k_copy(a_ptr, o_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr + off, tl.load(a_ptr + off, mask=m, other=0.0), mask=m)


@triton.jit
def k_flat8(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    c = off % 3
    mp1 = mask & (off + 1 < n)
    mp2 = mask & (off + 2 < n)
    mm1 = mask & (off > 0)
    mm2 = mask & (off > 1)
    ap1 = tl.load(a_ptr + off + 1, mask=mp1, other=0.0)
    ap2 = tl.load(a_ptr + off + 2, mask=mp2, other=0.0)
    am1 = tl.load(a_ptr + off - 1, mask=mm1, other=0.0)
    am2 = tl.load(a_ptr + off - 2, mask=mm2, other=0.0)
    bp1 = tl.load(b_ptr + off + 1, mask=mp1, other=0.0)
    bp2 = tl.load(b_ptr + off + 2, mask=mp2, other=0.0)
    bm1 = tl.load(b_ptr + off - 1, mask=mm1, other=0.0)
    bm2 = tl.load(b_ptr + off - 2, mask=mm2, other=0.0)
    is0 = c == 0
    is2 = c == 2
    a1 = tl.where(is2, am2, ap1)
    a2 = tl.where(is0, ap2, am1)
    b1 = tl.where(is2, bm2, bp1)
    b2 = tl.where(is0, bp2, bm1)
    tl.store(o_ptr + off, a1 * b2 - a2 * b1, mask=mask)


@triton.jit
def k_flat8aligned(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    """Eight loads, all at the same 32B-aligned offset (perf probe only)."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    ap1 = tl.load(a_ptr + off, mask=mask, other=0.0)
    ap2 = tl.load(b_ptr + off, mask=mask, other=0.0)
    am1 = tl.load(a_ptr + off, mask=mask, other=0.0)
    am2 = tl.load(b_ptr + off, mask=mask, other=0.0)
    bp1 = tl.load(a_ptr + off, mask=mask, other=0.0)
    bp2 = tl.load(b_ptr + off, mask=mask, other=0.0)
    bm1 = tl.load(a_ptr + off, mask=mask, other=0.0)
    bm2 = tl.load(b_ptr + off, mask=mask, other=0.0)
    r = (ap1 * ap2 - am1 * am2) + (bp1 * bp2 - bm1 * bm2)
    tl.store(o_ptr + off, r, mask=mask)


@triton.jit
def k_gather3(a_ptr, b_ptr, o_ptr, nrow, BLOCK: tl.constexpr):
    """(outer, 3, inner=1): three stride-3 slice loads per operand."""
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = r < nrow
    base = r * 3
    a0 = tl.load(a_ptr + base, mask=m, other=0.0)
    a1 = tl.load(a_ptr + base + 1, mask=m, other=0.0)
    a2 = tl.load(a_ptr + base + 2, mask=m, other=0.0)
    b0 = tl.load(b_ptr + base, mask=m, other=0.0)
    b1 = tl.load(b_ptr + base + 1, mask=m, other=0.0)
    b2 = tl.load(b_ptr + base + 2, mask=m, other=0.0)
    tl.store(o_ptr + base, a1 * b2 - a2 * b1, mask=m)
    tl.store(o_ptr + base + 1, a2 * b0 - a0 * b2, mask=m)
    tl.store(o_ptr + base + 2, a0 * b1 - a1 * b0, mask=m)


def bench(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    res = {"shape": [NROW, 3], "results": {}}
    o32 = torch.empty(N, device=DEV, dtype=torch.float32)
    a32 = torch.randn(N, device=DEV, dtype=torch.float32)
    b32 = torch.randn(N, device=DEV, dtype=torch.float32)
    o16 = torch.empty(N, device=DEV, dtype=torch.float16)
    a16 = a32.half()
    b16 = b32.half()

    def add(name, ms, note=""):
        res["results"][name] = {"ms": ms, "note": note}
        print(f"{name:26s} {ms:9.4f} ms  {note}")

    add("null_grid4", bench(lambda: k_null[(4,)](o32)), "host+launch overhead")

    for blk in (512, 2048, 4096):
        add(
            f"copy_f32_b{blk}",
            bench(lambda: k_copy[(triton.cdiv(N, blk),)](a32, o32, N, BLOCK=blk)),
            "1 load + 1 store, aligned",
        )

    for blk in (1024, 2048):
        add(
            f"flat8_f32_b{blk}",
            bench(lambda: k_flat8[(triton.cdiv(N, blk),)](a32, b32, o32, N, BLOCK=blk)),
            "current candidate: 8 shifted loads",
        )

    add(
        "flat8aligned_f32_b2048",
        bench(lambda: k_flat8aligned[(triton.cdiv(N, 2048),)](a32, b32, o32, N, BLOCK=2048)),
        "8 loads, all aligned (perf probe only)",
    )

    for blk in (512, 1024):
        add(
            f"gather3_f32_b{blk}",
            bench(lambda: k_gather3[(triton.cdiv(NROW, blk),)](a32, b32, o32, NROW, BLOCK=blk)),
            "6 stride-3 loads + 3 strides-3 stores",
        )

    add(
        "flat8_f16_b2048",
        bench(lambda: k_flat8[(triton.cdiv(N, 2048),)](a16, b16, o16, N, BLOCK=2048)),
        "current candidate, fp16",
    )
    add(
        "copy_f16_b2048",
        bench(lambda: k_copy[(triton.cdiv(N, 2048),)](a16, o16, N, BLOCK=2048)),
        "1 load + 1 store, aligned, fp16",
    )

    out_dir = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "probe.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("WROTE", os.path.join(out_dir, "probe.json"))


main()
