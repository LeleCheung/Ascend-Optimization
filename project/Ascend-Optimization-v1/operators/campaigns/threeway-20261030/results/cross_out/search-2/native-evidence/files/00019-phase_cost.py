"""Price the phase-mask machinery of cross_out's flat kernel.

Baseline: copy (2 aligned loads + add) and fwd4 (4 misaligned loads) are both
~0.068 ms at 1M vectors. fwd4 + `lane % 3` + 2 selects is 2.18 ms. This script
separates the modulo from the select and tries modulo-free phase masks.
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
    a = tl.load(a_ptr + off)
    b = tl.load(b_ptr + off)
    tl.store(o_ptr + off, a + b)


@triton.jit
def k_fwd4(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    base = tl.program_id(0) * STEP + lane
    a1 = tl.load(a_ptr + base + 1)
    a2 = tl.load(a_ptr + base + 2)
    b1 = tl.load(b_ptr + base + 1)
    b2 = tl.load(b_ptr + base + 2)
    tl.store(o_ptr + base, a1 * b2 - a2 * b1)


@triton.jit
def k_mod_only(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    off = tl.program_id(0) * STEP + lane
    a = tl.load(a_ptr + off)
    b = tl.load(b_ptr + off)
    ph = lane % 3
    tl.store(o_ptr + off, a + b + ph.to(tl.float32))


@triton.jit
def k_and_where(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    off = tl.program_id(0) * STEP + lane
    a = tl.load(a_ptr + off)
    b = tl.load(b_ptr + off)
    tl.store(o_ptr + off, tl.where((lane & 1) == 0, a, b))


@triton.jit
def k_mod_where(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    off = tl.program_id(0) * STEP + lane
    a = tl.load(a_ptr + off)
    b = tl.load(b_ptr + off)
    tl.store(o_ptr + off, tl.where((lane % 3) == 0, a, b))


@triton.jit
def k_floor_where(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    off = tl.program_id(0) * STEP + lane
    a = tl.load(a_ptr + off)
    b = tl.load(b_ptr + off)
    q = (lane.to(tl.float32) * 0.3333333432674408).to(tl.int32)
    ph = lane - 3 * q
    tl.store(o_ptr + off, tl.where(ph == 0, a, b))


@triton.jit
def k_floor_full(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    """Correct cross product; phase from fp32 truncation, two selects."""
    lane = tl.arange(0, BLOCK)
    base = tl.program_id(0) * STEP + lane
    q = (lane.to(tl.float32) * 0.3333333432674408).to(tl.int32)
    ph = lane - 3 * q
    a_m2 = tl.load(a_ptr + base - 2)
    a_m1 = tl.load(a_ptr + base - 1)
    a_p1 = tl.load(a_ptr + base + 1)
    a_p2 = tl.load(a_ptr + base + 2)
    b_m2 = tl.load(b_ptr + base - 2)
    b_m1 = tl.load(b_ptr + base - 1)
    b_p1 = tl.load(b_ptr + base + 1)
    b_p2 = tl.load(b_ptr + base + 2)
    t0 = a_p1 * b_p2 - a_p2 * b_p1
    t1 = a_p1 * b_m1 - a_m1 * b_p1
    t2 = a_m2 * b_m1 - a_m1 * b_m2
    tl.store(o_ptr + base, tl.where(ph == 0, t0, tl.where(ph == 1, t1, t2)))


@triton.jit
def k_mod_full(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    """Correct cross product; phase from integer modulo (expected slow path)."""
    lane = tl.arange(0, BLOCK)
    base = tl.program_id(0) * STEP + lane
    ph = lane % 3
    a_m2 = tl.load(a_ptr + base - 2)
    a_m1 = tl.load(a_ptr + base - 1)
    a_p1 = tl.load(a_ptr + base + 1)
    a_p2 = tl.load(a_ptr + base + 2)
    b_m2 = tl.load(b_ptr + base - 2)
    b_m1 = tl.load(b_ptr + base - 1)
    b_p1 = tl.load(b_ptr + base + 1)
    b_p2 = tl.load(b_ptr + base + 2)
    t0 = a_p1 * b_p2 - a_p2 * b_p1
    t1 = a_p1 * b_m1 - a_m1 * b_p1
    t2 = a_m2 * b_m1 - a_m1 * b_m2
    tl.store(o_ptr + base, tl.where(ph == 0, t0, tl.where(ph == 1, t1, t2)))


@triton.jit
def k_nophase(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    """Same 8 loads and 9 arithmetic ops, phase select removed: upper bound."""
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
    t0 = a_p1 * b_p2 - a_p2 * b_p1
    t1 = a_p1 * b_m1 - a_m1 * b_p1
    t2 = a_m2 * b_m1 - a_m1 * b_m2
    tl.store(o_ptr + base, t0 + t1 + t2)


def bench(fn, iters, warmup=8):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


TESTS = ["copy", "fwd4", "mod_only", "and_where", "mod_where", "floor_where",
         "nophase", "floor_full", "mod_full"]
CASES = [("fp16_1M", torch.float16, 1048576, 30), ("fp16_1024", torch.float16, 1024, 40)]


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
        block = 2048
        step = (block // 3) * 3
        g = ((n + step - 1) // step,)
        row = {}
        for kname in TESTS:
            kern = globals()["k_" + kname]
            entry = {}
            fn = (lambda kern=kern: kern[g](a, b, out, n, BLOCK=block, STEP=step))
            try:
                out.zero_()
                torch.npu.synchronize()
                fn()
                torch.npu.synchronize()
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

    path = os.path.join(os.environ.get("KGS_DEBUG_ARTIFACTS", "."), "phase_cost.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("WROTE", path)


if __name__ == "__main__":
    main()
