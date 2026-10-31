"""Bounded variant sweep for cross_out flat (last-dim, contiguous) path.

Compares the seed gather kernel against shuffle-free formulations on the target.
Diagnostic evidence only; not an authoritative measurement.
"""
import json
import os
import time
import traceback

import torch
import torch_npu
import triton
import triton.language as tl

OUT = {}


# ------------------------------- variants -------------------------------- #
@triton.jit
def k_copy(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    off = tl.program_id(0) * STEP + lane
    mask = (lane < STEP) & (off < n_elem)
    a = tl.load(a_ptr + off, mask=mask, other=0.0)
    b = tl.load(b_ptr + off, mask=mask, other=0.0)
    tl.store(o_ptr + off, a + b, mask=mask)


@triton.jit
def k_gather1d(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    off = tl.program_id(0) * STEP + lane
    mask = (lane < STEP) & (off < n_elem)
    a = tl.load(a_ptr + off, mask=mask, other=0.0)
    b = tl.load(b_ptr + off, mask=mask, other=0.0)
    phase = lane % 3
    p = tl.minimum(tl.where(phase == 2, lane - 2, lane + 1), STEP - 1)
    q = tl.minimum(tl.where(phase == 0, lane + 2, lane - 1), STEP - 1)
    a1 = tl.gather(a, p, axis=0)
    a2 = tl.gather(a, q, axis=0)
    b1 = tl.gather(b, p, axis=0)
    b2 = tl.gather(b, q, axis=0)
    tl.store(o_ptr + off, a1 * b2 - a2 * b1, mask=mask)


@triton.jit
def k_shift8(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    base = tl.program_id(0) * STEP + lane
    live = lane < STEP
    m_p1 = live & (base + 1 < n_elem)
    m_p2 = live & (base + 2 < n_elem)
    m_m1 = live & (base - 1 >= 0)
    m_m2 = live & (base - 2 >= 0)
    a_m2 = tl.load(a_ptr + base - 2, mask=m_m2, other=0.0)
    a_m1 = tl.load(a_ptr + base - 1, mask=m_m1, other=0.0)
    a_p1 = tl.load(a_ptr + base + 1, mask=m_p1, other=0.0)
    a_p2 = tl.load(a_ptr + base + 2, mask=m_p2, other=0.0)
    b_m2 = tl.load(b_ptr + base - 2, mask=m_m2, other=0.0)
    b_m1 = tl.load(b_ptr + base - 1, mask=m_m1, other=0.0)
    b_p1 = tl.load(b_ptr + base + 1, mask=m_p1, other=0.0)
    b_p2 = tl.load(b_ptr + base + 2, mask=m_p2, other=0.0)
    ph = lane % 3
    a1 = tl.where(ph == 2, a_m2, a_p1)
    a2 = tl.where(ph == 0, a_p2, a_m1)
    b1 = tl.where(ph == 2, b_m2, b_p1)
    b2 = tl.where(ph == 0, b_p2, b_m1)
    tl.store(o_ptr + base, a1 * b2 - a2 * b1, mask=live & (base < n_elem))


@triton.jit
def k_tile2d_perm(a_ptr, b_ptr, o_ptr, vectors, BV: tl.constexpr):
    rows = tl.program_id(0) * BV + tl.arange(0, BV)
    cols = tl.arange(0, 3)
    m = rows < vectors
    base = rows[:, None] * 3
    s1 = (cols[None, :] + 1) % 3
    s2 = (cols[None, :] + 2) % 3
    a1 = tl.load(a_ptr + base + s1, mask=m[:, None], other=0.0)
    a2 = tl.load(a_ptr + base + s2, mask=m[:, None], other=0.0)
    b1 = tl.load(b_ptr + base + s1, mask=m[:, None], other=0.0)
    b2 = tl.load(b_ptr + base + s2, mask=m[:, None], other=0.0)
    tl.store(o_ptr + base + cols[None, :], a1 * b2 - a2 * b1, mask=m[:, None])


@triton.jit
def k_tile2d_reshape(a_ptr, b_ptr, o_ptr, vectors, BV: tl.constexpr):
    rows = tl.program_id(0) * BV + tl.arange(0, BV)
    cols = tl.arange(0, 3)
    m = rows < vectors
    base = rows[:, None] * 3
    a = tl.load(a_ptr + base + cols[None, :], mask=m[:, None], other=0.0)
    b = tl.load(b_ptr + base + cols[None, :], mask=m[:, None], other=0.0)
    idx1 = tl.broadcast_to(((cols + 1) % 3)[None, :], (BV, 3))
    idx2 = tl.broadcast_to(((cols + 2) % 3)[None, :], (BV, 3))
    a1 = tl.gather(a, idx1, axis=1)
    a2 = tl.gather(a, idx2, axis=1)
    b1 = tl.gather(b, idx1, axis=1)
    b2 = tl.gather(b, idx2, axis=1)
    tl.store(o_ptr + base + cols[None, :], a1 * b2 - a2 * b1, mask=m[:, None])


@triton.jit
def k_strided3(a_ptr, b_ptr, o_ptr, vectors, BV: tl.constexpr):
    rows = tl.program_id(0) * BV + tl.arange(0, BV)
    m = rows < vectors
    p = rows * 3
    a0 = tl.load(a_ptr + p, mask=m, other=0.0)
    a1 = tl.load(a_ptr + p + 1, mask=m, other=0.0)
    a2 = tl.load(a_ptr + p + 2, mask=m, other=0.0)
    b0 = tl.load(b_ptr + p, mask=m, other=0.0)
    b1 = tl.load(b_ptr + p + 1, mask=m, other=0.0)
    b2 = tl.load(b_ptr + p + 2, mask=m, other=0.0)
    tl.store(o_ptr + p, a1 * b2 - a2 * b1, mask=m)
    tl.store(o_ptr + p + 1, a2 * b0 - a0 * b2, mask=m)
    tl.store(o_ptr + p + 2, a0 * b1 - a1 * b0, mask=m)


# ------------------------------- harness -------------------------------- #
def bench(fn, iters, warmup=8):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def flat_launch(kern, a, b, out, n, block, step):
    grid = ((n + step - 1) // step,)
    kern[grid](a, b, out, n, BLOCK=block, STEP=step)


def tile_launch(kern, a, b, out, n, bv):
    vectors = n // 3
    grid = ((vectors + bv - 1) // bv,)
    kern[grid](a, b, out, vectors, BV=bv)


CASES = [
    ("fp16_1024", torch.float16, 1024, 40),
    ("fp16_1M", torch.float16, 1048576, 30),
    ("fp32_1M", torch.float32, 1048576, 30),
]


def main():
    dev = "npu:0"
    results = {}
    for name, dt, nvec, iters in CASES:
        n = nvec * 3
        torch.manual_seed(0)
        a = torch.randn(n, dtype=dt, device=dev)
        b = torch.randn(n, dtype=dt, device=dev)
        out = torch.empty_like(a)
        ref = torch.cross(a.view(nvec, 3), b.view(nvec, 3), dim=-1).reshape(-1)
        row = {}

        block = 1024 if dt == torch.float32 else 2048
        step = (block // 3) * 3
        bv = 512

        for kname, fn in (
            ("copy", lambda: flat_launch(k_copy, a, b, out, n, block, step)),
            ("gather1d", lambda: flat_launch(k_gather1d, a, b, out, n, block, step)),
            ("shift8", lambda: flat_launch(k_shift8, a, b, out, n, block, step)),
            ("tile2d_perm", lambda: tile_launch(k_tile2d_perm, a, b, out, n, bv)),
            ("tile2d_reshape", lambda: tile_launch(k_tile2d_reshape, a, b, out, n, bv)),
            ("strided3", lambda: tile_launch(k_strided3, a, b, out, n, bv)),
        ):
            entry = {}
            try:
                out.zero_()
                torch.npu.synchronize()
                fn()
                torch.npu.synchronize()
                if kname != "copy":
                    err = (out.float() - ref.float()).abs().max().item()
                    entry["max_abs_err"] = err
                    entry["correct"] = bool(err < 0.02)
                entry["ms"] = bench(fn, iters)
            except Exception:
                entry["error"] = traceback.format_exc().splitlines()[-1]
            row[kname] = entry

        row["torch_cross_out"] = {"ms": bench(
            lambda: torch.cross(a.view(nvec, 3), b.view(nvec, 3), dim=-1, out=out.view(nvec, 3)),
            iters)}
        results[name] = row
        print(name, json.dumps(row), flush=True)

    path = os.path.join(os.environ.get("KGS_DEBUG_ARTIFACTS", "."), "variant_sweep.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("WROTE", path)


if __name__ == "__main__":
    main()
