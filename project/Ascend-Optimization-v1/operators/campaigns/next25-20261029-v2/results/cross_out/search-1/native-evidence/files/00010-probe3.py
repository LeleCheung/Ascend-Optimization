"""Probe #3: (a) load-count / UB-pressure threshold for shifted loads,
(b) Ascend CompiledKernel launch signature for low-overhead dispatch."""
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
def k_off2(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    """4 shifted loads, 2 distinct offsets per operand, no modulo/select."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    ap1 = tl.load(a_ptr + off + 1, mask=m, other=0.0)
    bp2 = tl.load(b_ptr + off + 2, mask=m, other=0.0)
    ap2 = tl.load(a_ptr + off + 2, mask=m, other=0.0)
    bp1 = tl.load(b_ptr + off + 1, mask=m, other=0.0)
    tl.store(o_ptr + off, ap1 * bp2 - ap2 * bp1, mask=m)


@triton.jit
def k_off3(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    """6 shifted loads, 3 distinct offsets per operand."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    ap1 = tl.load(a_ptr + off + 1, mask=m, other=0.0)
    bp2 = tl.load(b_ptr + off + 2, mask=m, other=0.0)
    ap2 = tl.load(a_ptr + off + 2, mask=m, other=0.0)
    bp1 = tl.load(b_ptr + off + 1, mask=m, other=0.0)
    am1 = tl.load(a_ptr + off - 1, mask=m, other=0.0)
    bm1 = tl.load(b_ptr + off - 1, mask=m, other=0.0)
    r = (ap1 * bp2 - ap2 * bp1) + (ap1 * bm1 - am1 * bp1)
    tl.store(o_ptr + off, r, mask=m)


@triton.jit
def k_off4(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    """8 shifted loads, 4 distinct offsets per operand, no modulo/select."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    ap1 = tl.load(a_ptr + off + 1, mask=m, other=0.0)
    bp2 = tl.load(b_ptr + off + 2, mask=m, other=0.0)
    ap2 = tl.load(a_ptr + off + 2, mask=m, other=0.0)
    bp1 = tl.load(b_ptr + off + 1, mask=m, other=0.0)
    am1 = tl.load(a_ptr + off - 1, mask=m, other=0.0)
    bm2 = tl.load(b_ptr + off - 2, mask=m, other=0.0)
    am2 = tl.load(a_ptr + off - 2, mask=m, other=0.0)
    bm1 = tl.load(b_ptr + off - 1, mask=m, other=0.0)
    r = (ap1 * bp2 - ap2 * bp1) + (am1 * bm2 - am2 * bm1)
    tl.store(o_ptr + off, r, mask=m)


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


def bench(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def probe(name, fn, note=""):
    try:
        ms = bench(fn)
        RES["results"][name] = {"ms": ms, "note": note}
        print(f"{name:26s} {ms:9.4f} ms  {note}")
    except Exception as e:  # noqa: BLE001
        RES["results"][name] = {"error": f"{type(e).__name__}: {e}"[:300]}
        print(f"{name:26s} ERROR {str(e)[:160]}")


def show_src(obj, needles, key):
    import inspect

    try:
        src = inspect.getsource(obj)
    except Exception as e:  # noqa: BLE001
        RES["info"][key] = {"error": str(e)[:200]}
        return
    lines = src.splitlines()
    keep = []
    for i, ln in enumerate(lines):
        if any(n in ln for n in needles):
            keep.extend(lines[max(0, i - 3) : i + 14])
    RES["info"][key] = "\n".join(keep)[:6000]
    print("=== " + key + " ===")
    print(RES["info"][key])


def main():
    a32 = torch.randn(N, device=DEV, dtype=torch.float32)
    b32 = torch.randn(N, device=DEV, dtype=torch.float32)
    o32 = torch.empty(N, device=DEV, dtype=torch.float32)
    g = lambda blk: (triton.cdiv(N, blk),)  # noqa: E731

    probe("copy_aligned_b2048", lambda: k_copy[g(2048)](a32, o32, N, BLOCK=2048))
    probe("off2_b2048", lambda: k_off2[g(2048)](a32, b32, o32, N, BLOCK=2048), "4 loads")
    probe("off3_b2048", lambda: k_off3[g(2048)](a32, b32, o32, N, BLOCK=2048), "6 loads")
    probe("off4_b2048", lambda: k_off4[g(2048)](a32, b32, o32, N, BLOCK=2048), "8 loads")
    probe("off4_b512", lambda: k_off4[g(512)](a32, b32, o32, N, BLOCK=512), "8 loads")
    probe("off4_b256", lambda: k_off4[g(256)](a32, b32, o32, N, BLOCK=256), "8 loads")
    probe("flat8_b256", lambda: k_flat8[g(256)](a32, b32, o32, N, BLOCK=256))
    probe("flat8_b64", lambda: k_flat8[g(64)](a32, b32, o32, N, BLOCK=64))

    # ---- launch signature ----
    try:
        import triton.spec.ascend.runtime.jit as ajit

        show_src(ajit.CompiledKernel, ["def __getitem__", "def run", "def _run"], "ascend_ck")
    except Exception as e:  # noqa: BLE001
        RES["info"]["ascend_ck_err"] = str(e)[:300]
        print("ck err", e)
    try:
        import inspect

        import triton.compiler.compiler as tcc

        RES["info"]["triton_ck_getitem"] = (
            inspect.getsource(tcc.CompiledKernel.__getitem__)[:2000]
        )
        print(RES["info"]["triton_ck_getitem"])
    except Exception as e:  # noqa: BLE001
        RES["info"]["triton_ck_getitem_err"] = str(e)[:300]

    out_dir = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "probe3.json"), "w") as f:
        json.dump(RES, f, indent=2, default=str)
    print("WROTE probe3.json")


main()
