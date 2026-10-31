"""Probe #4: isolate the 20x slowdown in the phase-selection kernel.

V1: shared mask + clamped indices, phase from `off % 3`      (non-constant mod)
V2: rows-per-program so base % 3 == 0 -> phase from `arange % 3` (constant mod)
V3: V2 without any modulo (selection by a fixed dummy) -> load/mask baseline
Plus correctness of V2 vs torch.cross for (1048576,3) fp32 and (1024,3) fp16,
and direct CompiledKernel launch timing.
"""
import json
import os
import time

import torch
import triton
import triton.language as tl

DEV = "npu"
RES = {"results": {}, "correct": {}, "info": {}}


@triton.jit
def k_v1(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    c = off % 3
    ip1 = tl.minimum(off + 1, n - 1)
    ip2 = tl.minimum(off + 2, n - 1)
    im1 = tl.maximum(off - 1, 0)
    im2 = tl.maximum(off - 2, 0)
    ap1 = tl.load(a_ptr + ip1, mask=mask, other=0.0)
    ap2 = tl.load(a_ptr + ip2, mask=mask, other=0.0)
    am1 = tl.load(a_ptr + im1, mask=mask, other=0.0)
    am2 = tl.load(a_ptr + im2, mask=mask, other=0.0)
    bp1 = tl.load(b_ptr + ip1, mask=mask, other=0.0)
    bp2 = tl.load(b_ptr + ip2, mask=mask, other=0.0)
    bm1 = tl.load(b_ptr + im1, mask=mask, other=0.0)
    bm2 = tl.load(b_ptr + im2, mask=mask, other=0.0)
    is0 = c == 0
    is2 = c == 2
    a1 = tl.where(is2, am2, ap1)
    a2 = tl.where(is0, ap2, am1)
    b1 = tl.where(is2, bm2, bp1)
    b2 = tl.where(is0, bp2, bm1)
    tl.store(o_ptr + off, a1 * b2 - a2 * b1, mask=mask)


@triton.jit
def k_v2(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr, STEP: tl.constexpr):
    ar = tl.arange(0, BLOCK)
    off = tl.program_id(0) * STEP + ar
    mask = (ar < STEP) & (off < n)
    c = ar % 3
    ip1 = tl.minimum(off + 1, n - 1)
    ip2 = tl.minimum(off + 2, n - 1)
    im1 = tl.maximum(off - 1, 0)
    im2 = tl.maximum(off - 2, 0)
    ap1 = tl.load(a_ptr + ip1, mask=mask, other=0.0)
    ap2 = tl.load(a_ptr + ip2, mask=mask, other=0.0)
    am1 = tl.load(a_ptr + im1, mask=mask, other=0.0)
    am2 = tl.load(a_ptr + im2, mask=mask, other=0.0)
    bp1 = tl.load(b_ptr + ip1, mask=mask, other=0.0)
    bp2 = tl.load(b_ptr + ip2, mask=mask, other=0.0)
    bm1 = tl.load(b_ptr + im1, mask=mask, other=0.0)
    bm2 = tl.load(b_ptr + im2, mask=mask, other=0.0)
    is0 = c == 0
    is2 = c == 2
    a1 = tl.where(is2, am2, ap1)
    a2 = tl.where(is0, ap2, am1)
    b1 = tl.where(is2, bm2, bp1)
    b2 = tl.where(is0, bp2, bm1)
    tl.store(o_ptr + off, a1 * b2 - a2 * b1, mask=mask)


@triton.jit
def k_v3(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr, STEP: tl.constexpr):
    """V2 shape but no modulo at all: measures load+mask overhead only."""
    ar = tl.arange(0, BLOCK)
    off = tl.program_id(0) * STEP + ar
    mask = (ar < STEP) & (off < n)
    ip1 = tl.minimum(off + 1, n - 1)
    ip2 = tl.minimum(off + 2, n - 1)
    im1 = tl.maximum(off - 1, 0)
    im2 = tl.maximum(off - 2, 0)
    ap1 = tl.load(a_ptr + ip1, mask=mask, other=0.0)
    ap2 = tl.load(a_ptr + ip2, mask=mask, other=0.0)
    am1 = tl.load(a_ptr + im1, mask=mask, other=0.0)
    am2 = tl.load(a_ptr + im2, mask=mask, other=0.0)
    bp1 = tl.load(b_ptr + ip1, mask=mask, other=0.0)
    bp2 = tl.load(b_ptr + ip2, mask=mask, other=0.0)
    bm1 = tl.load(b_ptr + im1, mask=mask, other=0.0)
    bm2 = tl.load(b_ptr + im2, mask=mask, other=0.0)
    a1 = tl.where(ap1 > 1e30, am2, ap1)
    a2 = tl.where(ap2 > 1e30, am1, ap2)
    b1 = tl.where(bp1 > 1e30, bm2, bp1)
    b2 = tl.where(bp2 > 1e30, bm1, bp2)
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
        print(f"{name:24s} {ms:9.4f} ms  {note}")
    except Exception as e:  # noqa: BLE001
        RES["results"][name] = {"error": f"{type(e).__name__}: {e}"[:300]}
        print(f"{name:24s} ERROR {str(e)[:160]}")


def step_of(blk):
    return blk - (blk % 3)


def run_v2(a, b, o, n, blk):
    k_v2[(triton.cdiv(n, step_of(blk)),)](a, b, o, n, BLOCK=blk, STEP=step_of(blk))


def check(tag, a, b, n, blk, dtype_tol):
    o = torch.empty(n, device=DEV, dtype=a.dtype)
    nrow = n // 3
    ref = torch.cross(a.view(nrow, 3), b.view(nrow, 3), dim=-1).reshape(n)
    run_v2(a, b, o, n, blk)
    torch.npu.synchronize()
    ok = torch.allclose(o, ref, rtol=dtype_tol[0], atol=dtype_tol[1], equal_nan=True)
    md = (o.float() - ref.float()).abs().max().item()
    RES["correct"][tag] = {"allclose": bool(ok), "max_abs_diff": md}
    print(f"correct {tag}: allclose={ok} maxdiff={md}")


def main():
    N = 3 * 1048576
    a32 = torch.randn(N, device=DEV, dtype=torch.float32)
    b32 = torch.randn(N, device=DEV, dtype=torch.float32)
    o32 = torch.empty(N, device=DEV, dtype=torch.float32)

    for blk in (2048, 512):
        st = step_of(blk)
        g = (triton.cdiv(N, st),)
        probe(
            f"v1_modoff_b{blk}",
            lambda blk=blk, st=st, g=g: k_v1[g](a32, b32, o32, N, BLOCK=blk),
        )
        probe(
            f"v2_modarange_b{blk}",
            lambda blk=blk, st=st, g=g: k_v2[g](a32, b32, o32, N, BLOCK=blk, STEP=st),
        )
        probe(
            f"v3_nomod_b{blk}",
            lambda blk=blk, st=st, g=g: k_v3[g](a32, b32, o32, N, BLOCK=blk, STEP=st),
        )

    check("f32_1M", a32, b32, N, 2048, (1e-4, 1e-4))
    Ns = 3 * 1024
    a16 = torch.randn(Ns, device=DEV, dtype=torch.float16)
    b16 = torch.randn(Ns, device=DEV, dtype=torch.float16)
    check("f16_1024_b512", a16, b16, Ns, 512, (1.5e-2, 1.5e-2))
    check("f16_1024_b256", a16, b16, Ns, 256, (1.5e-2, 1.5e-2))
    Nm = 3 * 65536
    am = torch.randn(Nm, device=DEV, dtype=torch.bfloat16)
    bm = torch.randn(Nm, device=DEV, dtype=torch.bfloat16)
    check("bf16_65536_b2048", am, bm, Nm, 2048, (1.5e-2, 1.5e-2))

    # --- direct launch path ---
    try:
        blk = 2048
        st = step_of(blk)
        g = (triton.cdiv(N, st), 1, 1)
        kern = k_v2.warmup(
            a32, b32, o32, N, BLOCK=blk, STEP=st, grid=g[:1]
        )
        runner = kern[g]
        probe("v2_direct_runner", lambda: runner(a32, b32, o32, N), "pre-bound")
        RES["info"]["runner_ok"] = True
    except Exception as e:  # noqa: BLE001
        RES["info"]["runner_err"] = f"{type(e).__name__}: {e}"[:400]
        print("runner err", e)

    # small-case launch overhead with the direct path
    probe("jitpath_f16_1024", lambda: run_v2(a16, b16, torch.empty(Ns, device=DEV, dtype=torch.float16), Ns, 512))

    out_dir = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "probe4.json"), "w") as f:
        json.dump(RES, f, indent=2, default=str)
    print("WROTE probe4.json")


main()
