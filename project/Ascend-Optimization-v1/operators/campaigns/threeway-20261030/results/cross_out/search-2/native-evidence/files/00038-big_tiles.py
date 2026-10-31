"""Price wider tiles and bf16 arithmetic for the three 1M-vector cases.

Uses the submitted candidate's own flat kernel so the measurement is of the real
code path, plus one local variant that does bf16 math in fp32.
Diagnostic evidence only.
"""
import importlib.util
import json
import os
import time
import traceback

import torch
import torch_npu
import triton
import triton.language as tl


def load_candidate():
    spec = importlib.util.spec_from_file_location("cand_main", "tmp/main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@triton.jit
def k_fp32math(a_ptr, b_ptr, o_ptr, n_elem, BLOCK: tl.constexpr, STEP: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    base = tl.program_id(0) * STEP + lane
    live = lane < STEP
    ph = lane - 3 * (lane.to(tl.float32) * 0.3333333432674408).to(tl.int32)
    lo2 = live & (base >= 2) & (base < n_elem)
    lo1 = live & (base >= 1) & (base < n_elem)
    hi1 = live & (base + 1 < n_elem)
    hi2 = live & (base + 2 < n_elem)
    a_m2 = tl.load(a_ptr + base - 2, mask=lo2, other=0.0).to(tl.float32)
    a_m1 = tl.load(a_ptr + base - 1, mask=lo1, other=0.0).to(tl.float32)
    a_p1 = tl.load(a_ptr + base + 1, mask=hi1, other=0.0).to(tl.float32)
    a_p2 = tl.load(a_ptr + base + 2, mask=hi2, other=0.0).to(tl.float32)
    b_m2 = tl.load(b_ptr + base - 2, mask=lo2, other=0.0).to(tl.float32)
    b_m1 = tl.load(b_ptr + base - 1, mask=lo1, other=0.0).to(tl.float32)
    b_p1 = tl.load(b_ptr + base + 1, mask=hi1, other=0.0).to(tl.float32)
    b_p2 = tl.load(b_ptr + base + 2, mask=hi2, other=0.0).to(tl.float32)
    t0 = a_p1 * b_p2 - a_p2 * b_p1
    t1 = a_p1 * b_m1 - a_m1 * b_p1
    t2 = a_m2 * b_m1 - a_m1 * b_m2
    r = tl.where(ph == 0, t0, tl.where(ph == 1, t1, t2))
    tl.store(o_ptr + base, r.to(o_ptr.dtype.element_ty),
             mask=live & (base < n_elem))


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
    ("fp16_1M", torch.float16, 1048576, (4096, 8192), 30),
    ("bf16_1M", torch.bfloat16, 1048576, (4096, 8192), 30),
    ("fp32_1M", torch.float32, 1048576, (2048, 4096), 30),
]


def main():
    m = load_candidate()
    kern = m._cross_flat_kernel
    dev = "npu:0"
    results = {}
    for name, dt, nvec, blocks, iters in CASES:
        n = nvec * 3
        torch.manual_seed(0)
        a = torch.randn(n, dtype=dt, device=dev)
        b = torch.randn(n, dtype=dt, device=dev)
        out = torch.empty(n, dtype=dt, device=dev)
        ref = torch.cross(a.view(nvec, 3), b.view(nvec, 3), dim=-1).reshape(-1)
        row = {}
        for blk in blocks:
            step = (blk // 3) * 3
            g = ((n + step - 1) // step,)
            entry = {}
            fn = (lambda blk=blk, step=step, g=g:
                  kern[g](a, b, out, n, BLOCK=blk, STEP=step))
            try:
                out.zero_()
                torch.npu.synchronize()
                fn()
                torch.npu.synchronize()
                entry["max_abs_err"] = (out.float() - ref.float()).abs().max().item()
                entry["grid"] = g[0]
                entry["ms"] = bench(fn, iters)
            except Exception:
                entry["error"] = traceback.format_exc().splitlines()[-1]
            row["cand_b%d" % blk] = entry
        if dt is torch.bfloat16:
            step = 4095
            g = ((n + step - 1) // step,)
            entry = {}
            fn = (lambda: k_fp32math[g](a, b, out, n, BLOCK=4096, STEP=4095))
            try:
                out.zero_()
                torch.npu.synchronize()
                fn()
                torch.npu.synchronize()
                entry["max_abs_err"] = (out.float() - ref.float()).abs().max().item()
                entry["ms"] = bench(fn, iters)
            except Exception:
                entry["error"] = traceback.format_exc().splitlines()[-1]
            row["fp32math_b4096"] = entry
        row["torch_cross_out"] = {"ms": bench(
            lambda: torch.cross(a.view(nvec, 3), b.view(nvec, 3), dim=-1, out=out.view(nvec, 3)),
            iters)}
        results[name] = row
        print(name, json.dumps(row), flush=True)

    path = os.path.join(os.environ.get("KGS_DEBUG_ARTIFACTS", "."), "big_tiles.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("WROTE", path)


if __name__ == "__main__":
    main()
