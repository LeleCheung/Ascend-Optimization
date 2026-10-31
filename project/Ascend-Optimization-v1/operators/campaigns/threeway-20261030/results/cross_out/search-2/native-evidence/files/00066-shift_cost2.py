"""Test whether the *form* of the offset expression (a_ptr + base - 2 versus
(a_ptr - 2) + base) determines whether cross_out's flat kernel vectorizes.

Diagnostic evidence only.
"""
import json
import os
import time
import traceback

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def k_copy(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    off = tl.program_id(0) * STEP + lane
    mask = (lane < STEP) & (off < n_elem)
    a = tl.load(a_ptr + off, mask=mask, other=0.0)
    b = tl.load(b_ptr + off, mask=mask, other=0.0)
    tl.store(o_ptr + off, a + b, mask=mask)


@triton.jit
def k_fwd4(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    """4 forward-only loads, no phase select (not the cross product)."""
    lane = tl.arange(0, BLOCK)
    base = tl.program_id(0) * STEP + lane
    a1 = tl.load(a_ptr + base + 1)
    a2 = tl.load(a_ptr + base + 2)
    b1 = tl.load(b_ptr + base + 1)
    b2 = tl.load(b_ptr + base + 2)
    tl.store(o_ptr + base, a1 * b2 - a2 * b1)


@triton.jit
def k_fwd4_where(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    """4 forward loads plus phase modulo/select, to price the select network."""
    lane = tl.arange(0, BLOCK)
    base = tl.program_id(0) * STEP + lane
    a1 = tl.load(a_ptr + base + 1)
    a2 = tl.load(a_ptr + base + 2)
    b1 = tl.load(b_ptr + base + 1)
    b2 = tl.load(b_ptr + base + 2)
    ph = lane % 3
    tl.store(o_ptr + base, tl.where(ph == 0, a1 * b2 - a2 * b1, tl.where(ph == 1, a2 * b1, a1 * b2)))


@triton.jit
def k_idx8(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    """Correct formula, backward streams via index arithmetic base-1/base-2."""
    lane = tl.arange(0, BLOCK)
    base = tl.program_id(0) * STEP + lane
    a_m2 = tl.load(a_ptr + base - 2, mask=base >= 2, other=0.0)
    a_m1 = tl.load(a_ptr + base - 1, mask=base >= 1, other=0.0)
    a_p1 = tl.load(a_ptr + base + 1)
    a_p2 = tl.load(a_ptr + base + 2)
    b_m2 = tl.load(b_ptr + base - 2, mask=base >= 2, other=0.0)
    b_m1 = tl.load(b_ptr + base - 1, mask=base >= 1, other=0.0)
    b_p1 = tl.load(b_ptr + base + 1)
    b_p2 = tl.load(b_ptr + base + 2)
    ph = lane % 3
    a1 = tl.where(ph == 2, a_m2, a_p1)
    a2 = tl.where(ph == 0, a_p2, a_m1)
    b1 = tl.where(ph == 2, b_m2, b_p1)
    b2 = tl.where(ph == 0, b_p2, b_m1)
    tl.store(o_ptr + base, a1 * b2 - a2 * b1)


@triton.jit
def k_inptr8(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    """Same 8 streams, but the constant lives on the scalar pointer."""
    lane = tl.arange(0, BLOCK)
    base = tl.program_id(0) * STEP + lane
    am2 = a_ptr - 2
    am1 = a_ptr - 1
    bm2 = b_ptr - 2
    bm1 = b_ptr - 1
    a_m2 = tl.load(am2 + base)
    a_m1 = tl.load(am1 + base)
    a_p1 = tl.load(a_ptr + base + 1)
    a_p2 = tl.load(a_ptr + base + 2)
    b_m2 = tl.load(bm2 + base)
    b_m1 = tl.load(bm1 + base)
    b_p1 = tl.load(b_ptr + base + 1)
    b_p2 = tl.load(b_ptr + base + 2)
    ph = lane % 3
    a1 = tl.where(ph == 2, a_m2, a_p1)
    a2 = tl.where(ph == 0, a_p2, a_m1)
    b1 = tl.where(ph == 2, b_m2, b_p1)
    b2 = tl.where(ph == 0, b_p2, b_m1)
    tl.store(o_ptr + base, a1 * b2 - a2 * b1)


@triton.jit
def k_inptr8_masked(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    """Pointer-hoisted backward streams, with the full production-style masking."""
    lane = tl.arange(0, BLOCK)
    base = tl.program_id(0) * STEP + lane
    live = lane < STEP
    hi = base < n_elem
    lo2 = base >= 2
    lo1 = base >= 1
    am2 = a_ptr - 2
    am1 = a_ptr - 1
    bm2 = b_ptr - 2
    bm1 = b_ptr - 1
    a_m2 = tl.load(am2 + base, mask=live & lo2, other=0.0)
    a_m1 = tl.load(am1 + base, mask=live & lo1, other=0.0)
    a_p1 = tl.load(a_ptr + base + 1, mask=live & hi, other=0.0)
    a_p2 = tl.load(a_ptr + base + 2, mask=live & hi, other=0.0)
    b_m2 = tl.load(bm2 + base, mask=live & lo2, other=0.0)
    b_m1 = tl.load(bm1 + base, mask=live & lo1, other=0.0)
    b_p1 = tl.load(b_ptr + base + 1, mask=live & hi, other=0.0)
    b_p2 = tl.load(b_ptr + base + 2, mask=live & hi, other=0.0)
    ph = lane % 3
    a1 = tl.where(ph == 2, a_m2, a_p1)
    a2 = tl.where(ph == 0, a_p2, a_m1)
    b1 = tl.where(ph == 2, b_m2, b_p1)
    b2 = tl.where(ph == 0, b_p2, b_m1)
    tl.store(o_ptr + base, a1 * b2 - a2 * b1, mask=live & hi)


@triton.jit
def k_gather1d(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    off = tl.program_id(0) * STEP + lane
    mask = (lane < STEP) & (off < n_elem)
    a = tl.load(a_ptr + off, mask=mask, other=0.0)
    b = tl.load(b_ptr + off, mask=mask, other=0.0)
    ph = lane % 3
    p = tl.minimum(tl.where(ph == 2, lane - 2, lane + 1), STEP - 1)
    q = tl.minimum(tl.where(ph == 0, lane + 2, lane - 1), STEP - 1)
    a1 = tl.gather(a, p, axis=0)
    a2 = tl.gather(a, q, axis=0)
    b1 = tl.gather(b, p, axis=0)
    b2 = tl.gather(b, q, axis=0)
    tl.store(o_ptr + off, a1 * b2 - a2 * b1, mask=mask)


def bench(fn, iters, warmup=8):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


CASES = [
    ("fp16_1M", torch.float16, 1048576, 30),
    ("fp32_1M", torch.float32, 1048576, 30),
    ("fp16_1024", torch.float16, 1024, 40),
]


def main():
    dev = "npu:0"
    results = {}
    for name, dt, nvec, iters in CASES:
        n = nvec * 3
        pad = 64
        torch.manual_seed(0)
        a_pad = torch.randn(n + pad, dtype=dt, device=dev)
        b_pad = torch.randn(n + pad, dtype=dt, device=dev)
        o_pad = torch.empty(n + pad, dtype=dt, device=dev)
        a, b, out = a_pad[:n], b_pad[:n], o_pad[:n]
        ref = torch.cross(a.view(nvec, 3), b.view(nvec, 3), dim=-1).reshape(-1)

        block = 1024 if dt == torch.float32 else 2048
        step = (block // 3) * 3
        g = ((n + step - 1) // step,)
        row = {}

        tests = {
            "copy": k_copy,
            "fwd4": k_fwd4,
            "fwd4_where": k_fwd4_where,
            "idx8": k_idx8,
            "inptr8": k_inptr8,
            "inptr8_masked": k_inptr8_masked,
            "gather1d": k_gather1d,
        }

        for kname, kern in tests.items():
            entry = {}
            fn = (lambda kern=kern: kern[g](a, b, out, n, BLOCK=block, STEP=step))
            try:
                out.zero_()
                torch.npu.synchronize()
                fn()
                torch.npu.synchronize()
                if kname in ("idx8", "inptr8", "inptr8_masked", "gather1d"):
                    entry["max_abs_err"] = (out.float() - ref.float()).abs().max().item()
                entry["ms"] = bench(fn, iters)
            except Exception:
                entry["error"] = traceback.format_exc().splitlines()[-1]
            row[kname] = entry

        row["torch_cross_out"] = {"ms": bench(
            lambda: torch.cross(a.view(nvec, 3), b.view(nvec, 3), dim=-1, out=out.view(nvec, 3)),
            iters)}
        results[name] = row
        print(name, json.dumps(row), flush=True)

    path = os.path.join(os.environ.get("KGS_DEBUG_ARTIFACTS", "."), "shift_cost2.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("WROTE", path)


if __name__ == "__main__":
    main()
