"""Isolate the cost driver of cross_out's flat path on the target.

Compares a pure copy floor against: a 32B-aligned shifted load, masked vs
unmasked 1-element shifted loads, the full 8-shifted-load formula, and a
2D-tile (BV,3) load with masked-reduce column extraction.
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
def k_shift_aligned(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr,
                    SH: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    off = tl.program_id(0) * STEP + lane
    mask = (lane < STEP) & (off < n_elem)
    a = tl.load(a_ptr + off, mask=mask, other=0.0)
    # SH elements is a whole 32B unit, so the shifted load stays aligned.
    b = tl.load(b_ptr + off + SH, mask=mask, other=0.0)
    tl.store(o_ptr + off, a + b, mask=mask)


@triton.jit
def k_shift1_masked(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    off = tl.program_id(0) * STEP + lane
    mask = (lane < STEP) & (off < n_elem)
    a = tl.load(a_ptr + off, mask=mask, other=0.0)
    b = tl.load(b_ptr + off + 1, mask=mask, other=0.0)
    tl.store(o_ptr + off, a + b, mask=mask)


@triton.jit
def k_shift1_unmasked(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    off = tl.program_id(0) * STEP + lane
    a = tl.load(a_ptr + off)
    b = tl.load(b_ptr + off + 1)
    tl.store(o_ptr + off, a + b)


@triton.jit
def k_shift8_unmasked(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    base = tl.program_id(0) * STEP + lane
    a_m2 = tl.load(a_ptr + base - 2)
    a_m1 = tl.load(a_ptr + base - 1)
    a_p1 = tl.load(a_ptr + base + 1)
    a_p2 = tl.load(a_ptr + base + 2)
    b_m2 = tl.load(b_ptr + base - 2)
    b_m1 = tl.load(b_ptr + base - 1)
    b_p1 = tl.load(b_ptr + base + 1)
    b_p2 = tl.load(b_ptr + base + 2)
    ph = lane % 3
    a1 = tl.where(ph == 2, a_m2, a_p1)
    a2 = tl.where(ph == 0, a_p2, a_m1)
    b1 = tl.where(ph == 2, b_m2, b_p1)
    b2 = tl.where(ph == 0, b_p2, b_m1)
    tl.store(o_ptr + base, a1 * b2 - a2 * b1)


@triton.jit
def k_tile2d_reduce(a_ptr, b_ptr, o_ptr, vectors, BV: tl.constexpr):
    rows = tl.program_id(0) * BV + tl.arange(0, BV)
    cols = tl.arange(0, 3)
    m = rows < vectors
    base = rows[:, None] * 3
    a_t = tl.load(a_ptr + base + cols[None, :], mask=m[:, None], other=0.0)
    b_t = tl.load(b_ptr + base + cols[None, :], mask=m[:, None], other=0.0)
    z = tl.zeros_like(a_t)
    a0 = tl.sum(tl.where(cols[None, :] == 0, a_t, z), axis=1)
    a1 = tl.sum(tl.where(cols[None, :] == 1, a_t, z), axis=1)
    a2 = tl.sum(tl.where(cols[None, :] == 2, a_t, z), axis=1)
    b0 = tl.sum(tl.where(cols[None, :] == 0, b_t, z), axis=1)
    b1 = tl.sum(tl.where(cols[None, :] == 1, b_t, z), axis=1)
    b2 = tl.sum(tl.where(cols[None, :] == 2, b_t, z), axis=1)
    c0 = a1 * b2 - a2 * b1
    c1 = a2 * b0 - a0 * b2
    c2 = a0 * b1 - a1 * b0
    e0 = tl.where(cols == 0, 1.0, 0.0)
    e1 = tl.where(cols == 1, 1.0, 0.0)
    e2 = tl.where(cols == 2, 1.0, 0.0)
    out_t = c0[:, None] * e0[None, :] + c1[:, None] * e1[None, :] + c2[:, None] * e2[None, :]
    tl.store(o_ptr + base + cols[None, :], out_t, mask=m[:, None])


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
        full = (n // step) * step
        g_full = (full // step,)
        bv = 512
        gv = (nvec // bv,)
        row = {}

        def flat(kern, grid, **extra):
            return lambda: kern[grid](a, b, out, n, BLOCK=block, STEP=step, **extra)

        tests = {
            "copy": flat(k_copy, g_full),
            "shift_aligned32B": flat(k_shift_aligned, g_full, SH=(16 if dt != torch.float32 else 8)),
            "shift1_masked": flat(k_shift1_masked, g_full),
            "shift1_unmasked": flat(k_shift1_unmasked, g_full),
            "shift8_unmasked": flat(k_shift8_unmasked, g_full),
            "tile2d_reduce": lambda: k_tile2d_reduce[gv](a, b, out, nvec, BV=bv),
        }
        for kname, fn in tests.items():
            entry = {}
            try:
                out.zero_()
                torch.npu.synchronize()
                fn()
                torch.npu.synchronize()
                if kname in ("shift8_unmasked", "tile2d_reduce"):
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

    path = os.path.join(os.environ.get("KGS_DEBUG_ARTIFACTS", "."), "shift_cost.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("WROTE", path)


if __name__ == "__main__":
    main()
