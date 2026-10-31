"""Probe #2: is the failure (a) unaligned base address, (b) index-expression
shape, or (c) intrinsic?  Plus host-dispatch overhead anatomy.
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
RES = {"results": {}, "info": {}}


@triton.jit
def k_copy(a_ptr, o_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr + off, tl.load(a_ptr + off, mask=m, other=0.0), mask=m)


@triton.jit
def k_copy_shift1(a_ptr, o_ptr, n, BLOCK: tl.constexpr):
    """Aligned store, base address shifted by one element."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr + off, tl.load(a_ptr + off + 1, mask=m, other=0.0), mask=m)


@triton.jit
def k_copy_ptrshift1(a_ptr, o_ptr, n, BLOCK: tl.constexpr):
    """Same as above but the shift is folded into the pointer base."""
    ap = a_ptr + 1
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr + off, tl.load(ap + off, mask=m, other=0.0), mask=m)


@triton.jit
def k_copy_off2(a_ptr, o_ptr, n, BLOCK: tl.constexpr):
    """Load at off+2, i.e. 8-byte shift."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr + off, tl.load(a_ptr + off + 2, mask=m, other=0.0), mask=m)


@triton.jit
def k_flat8hint(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    c = off % 3
    ap1 = tl.load(a_ptr + tl.max_contiguous(tl.multiple_of(off + 1, 1), BLOCK), mask=mask, other=0.0)
    ap2 = tl.load(a_ptr + tl.max_contiguous(tl.multiple_of(off + 2, 1), BLOCK), mask=mask, other=0.0)
    am1 = tl.load(a_ptr + tl.max_contiguous(tl.multiple_of(off - 1, 1), BLOCK), mask=mask, other=0.0)
    am2 = tl.load(a_ptr + tl.max_contiguous(tl.multiple_of(off - 2, 1), BLOCK), mask=mask, other=0.0)
    bp1 = tl.load(b_ptr + tl.max_contiguous(tl.multiple_of(off + 1, 1), BLOCK), mask=mask, other=0.0)
    bp2 = tl.load(b_ptr + tl.max_contiguous(tl.multiple_of(off + 2, 1), BLOCK), mask=mask, other=0.0)
    bm1 = tl.load(b_ptr + tl.max_contiguous(tl.multiple_of(off - 1, 1), BLOCK), mask=mask, other=0.0)
    bm2 = tl.load(b_ptr + tl.max_contiguous(tl.multiple_of(off - 2, 1), BLOCK), mask=mask, other=0.0)
    is0 = c == 0
    is2 = c == 2
    a1 = tl.where(is2, am2, ap1)
    a2 = tl.where(is0, ap2, am1)
    b1 = tl.where(is2, bm2, bp1)
    b2 = tl.where(is0, bp2, bm1)
    tl.store(o_ptr + off, a1 * b2 - a2 * b1, mask=mask)


@triton.jit
def k_tile4(a_ptr, b_ptr, o_ptr, nrow, BLOCK: tl.constexpr):
    """2-D strided tile (BLOCK,4) covering rows of the 3-wide cross axis."""
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    col = tl.arange(0, 4)
    idx = rows[:, None] * 3 + col[None, :]
    mrow = rows < nrow
    m = mrow[:, None] & (col[None, :] < 3)
    A = tl.load(a_ptr + idx, mask=m, other=0.0)
    B = tl.load(b_ptr + idx, mask=m, other=0.0)
    c0 = (col == 0)[None, :]
    c1 = (col == 1)[None, :]
    c2 = (col == 2)[None, :]
    a0 = tl.sum(tl.where(c0, A, 0.0), 1)
    a1 = tl.sum(tl.where(c1, A, 0.0), 1)
    a2 = tl.sum(tl.where(c2, A, 0.0), 1)
    b0 = tl.sum(tl.where(c0, B, 0.0), 1)
    b1 = tl.sum(tl.where(c1, B, 0.0), 1)
    b2 = tl.sum(tl.where(c2, B, 0.0), 1)
    r0 = (a1 * b2 - a2 * b1)[:, None]
    r1 = (a2 * b0 - a0 * b2)[:, None]
    r2 = (a0 * b1 - a1 * b0)[:, None]
    out = tl.where(c0, r0, tl.where(c1, r1, r2))
    tl.store(o_ptr + idx, out, mask=m)


def bench(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def add(name, ms, note=""):
    RES["results"][name] = {"ms": ms, "note": note}
    print(f"{name:26s} {ms:9.4f} ms  {note}")


def probe(name, fn, note=""):
    try:
        add(name, bench(fn), note)
    except Exception as e:  # noqa: BLE001
        RES["results"][name] = {"error": f"{type(e).__name__}: {e}"}
        print(f"{name:26s} ERROR {type(e).__name__}: {str(e)[:200]}")


def main():
    a32 = torch.randn(N, device=DEV, dtype=torch.float32)
    b32 = torch.randn(N, device=DEV, dtype=torch.float32)
    o32 = torch.empty(N, device=DEV, dtype=torch.float32)
    B = 2048
    g = triton.cdiv(N, B)

    probe("copy_aligned", lambda: k_copy[(g,)](a32, o32, N, BLOCK=B))
    probe("copy_shift1_idx", lambda: k_copy_shift1[(g,)](a32, o32, N, BLOCK=B))
    probe("copy_shift1_ptr", lambda: k_copy_ptrshift1[(g,)](a32, o32, N, BLOCK=B))
    probe("copy_shift2_idx", lambda: k_copy_off2[(g,)](a32, o32, N, BLOCK=B))
    probe("flat8_hint", lambda: k_flat8hint[(g,)](a32, b32, o32, N, BLOCK=B))
    probe(
        "tile4_b256",
        lambda: k_tile4[(triton.cdiv(NROW, 256),)](a32, b32, o32, NROW, BLOCK=256),
    )

    # ---- host dispatch anatomy ----
    try:
        RES["info"]["jit_type"] = type(k_copy).__name__
        RES["info"]["jit_dir"] = [
            d for d in dir(k_copy) if not d.startswith("__")
        ]
        kern = k_copy.warmup(a32, o32, N, BLOCK=B, grid=(1,))
        RES["info"]["compiled_type"] = type(kern).__name__
        RES["info"]["compiled_dir"] = [d for d in dir(kern) if not d.startswith("__")]
        try:
            kern._init_handles()
        except Exception as e:  # noqa: BLE001
            RES["info"]["init_handles_err"] = str(e)[:200]
        cg = kern[(g, 1, 1)]
        probe("copy_aligned_direct", lambda: cg(a32, o32, N), "pre-bound CompiledKernel")
    except Exception as e:  # noqa: BLE001
        RES["info"]["direct_err"] = f"{type(e).__name__}: {e}"[:400]
        print("direct err", e)

    # host-only python cost of the wrapper arithmetic
    n_elem = N
    o16 = o32

    def pyonly():
        if n_elem:
            blk = 2048 if n_elem > 131072 else 1024
            triton.cdiv(n_elem, blk)

    probe("python_only", pyonly, "wrapper arithmetic, no launch")

    out_dir = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "probe2.json"), "w") as f:
        json.dump(RES, f, indent=2, default=str)
    print("WROTE probe2.json")


main()
