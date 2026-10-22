"""Test non-scan prefix-sum formulations: tl.dot with triangular matrix, and
broadcast+reduce segmented prefix. Target: replace tl.cumsum (~0.45 ns/elt).
"""
import json
import os
import time

import torch
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
res = {"cases": [], "errors": []}


@triton.jit
def k_dot_raw(y, o, N: tl.constexpr, BR: tl.constexpr, L: tl.constexpr):
    pr = tl.program_id(0)
    pc = tl.program_id(1)
    r = pr * BR + tl.arange(0, BR)
    c = pc * L + tl.arange(0, L)
    t = tl.load(y + r[:, None] * N + c[None, :])
    seg = (c[:, None] >= c[None, :]).to(t.dtype)
    out = tl.dot(t, seg)
    tl.store(o + r[:, None] * N + c[None, :], out)


@triton.jit
def k_dot_prefix(y, o, NOUT: tl.constexpr, N: tl.constexpr, BR: tl.constexpr,
                 L: tl.constexpr, SCALE: tl.constexpr):
    pr = tl.program_id(0)
    r = pr * BR + tl.arange(0, BR)
    c = tl.arange(0, L)
    seg = (c[:, None] >= c[None, :]).to(tl.float16)
    carry = tl.zeros((BR, L), dtype=tl.float32)
    for cc in range(0, N // L):
        kk = cc * L + c
        a = tl.load(y + r[:, None] * N + kk[None, :]).to(tl.float16)
        b = tl.load(y + r[:, None] * N + (kk + 1)[None, :]).to(tl.float16)
        t = (a + b) * SCALE
        lp = tl.dot(t, seg).to(tl.float32)
        lp = lp + carry
        tl.store(o + r[:, None] * NOUT + kk[None, :], lp)
        last = tl.sum(tl.where(c[None, :] == (L - 1), lp, 0.0), axis=1)
        carry = last[:, None] + tl.zeros((BR, L), dtype=tl.float32)


@triton.jit
def k_bc_prefix(y, o, NOUT: tl.constexpr, N: tl.constexpr, BR: tl.constexpr,
                L: tl.constexpr, SCALE: tl.constexpr):
    pr = tl.program_id(0)
    r = pr * BR + tl.arange(0, BR)
    c = tl.arange(0, L)
    seg = (c[:, None] >= c[None, :]).to(tl.float32)
    carry = tl.zeros((BR, L), dtype=tl.float32)
    for cc in range(0, N // L):
        kk = cc * L + c
        a = tl.load(y + r[:, None] * N + kk[None, :]).to(tl.float32)
        b = tl.load(y + r[:, None] * N + (kk + 1)[None, :]).to(tl.float32)
        t = (a + b) * SCALE
        lp = tl.sum(t[:, :, None] * seg[None, :, :], axis=1) + carry
        tl.store(o + r[:, None] * NOUT + kk[None, :], lp)
        last = tl.sum(tl.where(c[None, :] == (L - 1), lp, 0.0), axis=1)
        carry = last[:, None] + tl.zeros((BR, L), dtype=tl.float32)


def wall(fn, iters=30, warm=8):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def dev(fn, iters=30, warm=8):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    e0 = torch.npu.Event(enable_timing=True)
    e1 = torch.npu.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.npu.synchronize()
    return e0.elapsed_time(e1) / iters


def add(name, fn, out=None, ref=None):
    try:
        fn()
        torch.npu.synchronize()
        rec = {"case": name, "dev_ms": dev(fn), "wall_ms": wall(fn)}
        if out is not None and ref is not None:
            rec["maxabs"] = (out.float() - ref.float()).abs().max().item()
        res["cases"].append(rec)
    except Exception as e:
        res["errors"].append({"case": name, "err": repr(e)[-450:]})


def main():
    torch.npu.set_device(0)
    R, N = 1024, 4096
    for dt in ("float16", "float32"):
        tdt = getattr(torch, dt)
        y = torch.randn(R, N, generator=torch.Generator().manual_seed(1730)).to(tdt).npu()
        ref = torch.cumulative_trapezoid(y, dx=1.0, dim=-1)
        o = torch.empty(R, N, dtype=tdt, device="npu:0")
        on = torch.empty(R, N - 1, dtype=tdt, device="npu:0")
        # raw dot throughput (local prefix only, wrong globally)
        for BR, L in ((16, 64), (16, 128), (32, 128), (16, 256), (64, 64)):
            g = (R // BR, N // L)
            add(f"dotraw-{dt}-br{BR}-L{L}",
                lambda g=g, BR=BR, L=L, y=y, o=o: k_dot_raw[g](y, o, N=N, BR=BR, L=L))
        # full prefix with in-program chunk loop
        combos = ((16, 128), (32, 128), (16, 256), (8, 128)) if dt == "float16" else ((16, 128), (16, 256))
        for BR, L in combos:
            g = (R // BR,)
            add(f"dotpref-{dt}-br{BR}-L{L}",
                lambda g=g, BR=BR, L=L, y=y, o=on: k_dot_prefix[g](y, o, NOUT=N - 1, N=N, BR=BR, L=L, SCALE=0.5),
                on, ref)
        # broadcast+reduce variant
        for BR, L in ((16, 32), (16, 64), (32, 32)):
            g = (R // BR,)
            add(f"bcpref-{dt}-br{BR}-L{L}",
                lambda g=g, BR=BR, L=L, y=y, o=on: k_bc_prefix[g](y, o, NOUT=N - 1, N=N, BR=BR, L=L, SCALE=0.5),
                on, ref)
        res["cases"].append({"case": f"torchREF-{dt}-1024x4096",
                             "wall_ms": wall(lambda: torch.cumulative_trapezoid(y, dx=1.0, dim=-1))})
    with open(os.path.join(ART, "bench5.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1)[:24000])


if __name__ == "__main__":
    main()
