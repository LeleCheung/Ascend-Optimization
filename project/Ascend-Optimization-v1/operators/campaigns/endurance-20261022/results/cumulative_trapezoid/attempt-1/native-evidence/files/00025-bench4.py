"""Isolate the cost of tile load/store vs tl.cumsum vs tl.sum vs tl.dot on 910B.

All measurements on (1024,4096) and (128,1024) tiles; results are diagnostic
(numerically only the copy/rowsum variants are meaningful).
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
def k_copy(y, o, N: tl.constexpr, BR: tl.constexpr, L: tl.constexpr, MASKED: tl.constexpr):
    pr = tl.program_id(0)
    pc = tl.program_id(1)
    r = pr * BR + tl.arange(0, BR)
    c = pc * L + tl.arange(0, L)
    p = r[:, None] * N + c[None, :]
    if MASKED:
        m = (r[:, None] < NR) & (c[None, :] < NC)
        v = tl.load(y + p, mask=m, other=0.0)
        tl.store(o + p, v, mask=m)
    else:
        v = tl.load(y + p)
        tl.store(o + p, v)


@triton.jit
def k_csum(y, o, N: tl.constexpr, BR: tl.constexpr, L: tl.constexpr):
    pr = tl.program_id(0)
    pc = tl.program_id(1)
    r = pr * BR + tl.arange(0, BR)
    c = pc * L + tl.arange(0, L)
    v = tl.load(y + r[:, None] * N + c[None, :]).to(tl.float32)
    cs = tl.cumsum(v, axis=1)
    tl.store(o + r[:, None] * N + c[None, :], cs)


@triton.jit
def k_rowsum(y, o, N: tl.constexpr, BR: tl.constexpr, L: tl.constexpr):
    pr = tl.program_id(0)
    pc = tl.program_id(1)
    r = pr * BR + tl.arange(0, BR)
    c = pc * L + tl.arange(0, L)
    v = tl.load(y + r[:, None] * N + c[None, :]).to(tl.float32)
    s = tl.sum(v, axis=1)
    tl.store(o + r * 64 + pc, s)


@triton.jit
def k_dot(y, o, N: tl.constexpr, BR: tl.constexpr, L: tl.constexpr):
    pr = tl.program_id(0)
    pc = tl.program_id(1)
    r = pr * BR + tl.arange(0, BR)
    c = pc * L + tl.arange(0, L)
    v = tl.load(y + r[:, None] * N + c[None, :])
    tril = (c[:, None] >= c[None, :]).to(v.dtype)
    res = tl.dot(v, tril)
    tl.store(o + r[:, None] * N + c[None, :], res)


def wall(fn, iters=50, warm=10):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def dev(fn, iters=50, warm=10):
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


def add(name, fn):
    try:
        fn()
        torch.npu.synchronize()
        res["cases"].append({"case": name, "dev_ms": dev(fn), "wall_ms": wall(fn)})
    except Exception as e:
        res["errors"].append({"case": name, "err": repr(e)[-500:]})


def main():
    torch.npu.set_device(0)
    global NR, NC
    for dt, R, N in (("float16", 1024, 4096), ("float32", 1024, 4096), ("float16", 128, 1024)):
        NR, NC = R, N
        tdt = getattr(torch, dt)
        y = torch.randn(R, N, generator=torch.Generator().manual_seed(1729)).to(tdt).npu()
        o = torch.empty(R, N, dtype=tdt, device="npu:0")
        small = torch.empty(R * 64, dtype=torch.float32, device="npu:0")
        tag = f"{dt}-{R}x{N}"
        for BR, L in ((1, 1024), (2, 1024), (4, 1024), (8, 512), (16, 256), (8, 1024)):
            for masked in (0,):
                g = (triton.cdiv(R, BR), triton.cdiv(N, L))
                add(f"copy-{tag}-br{BR}-L{L}-m{masked}",
                    lambda g=g, BR=BR, L=L, masked=masked, y=y, o=o: k_copy[g](y, o, N=N, BR=BR, L=L, MASKED=masked))
            g = (triton.cdiv(R, BR), triton.cdiv(N, L))
            add(f"csum-{tag}-br{BR}-L{L}",
                lambda g=g, BR=BR, L=L, y=y, o=o: k_csum[g](y, o, N=N, BR=BR, L=L))
            add(f"rowsum-{tag}-br{BR}-L{L}",
                lambda g=g, BR=BR, L=L, y=y, o=small: k_rowsum[g](y, o, N=N, BR=BR, L=L))
            if L <= 256 and dt != "float32":
                add(f"dot-{tag}-br{BR}-L{L}",
                    lambda g=g, BR=BR, L=L, y=y, o=o: k_dot[g](y, o, N=N, BR=BR, L=L))
        res["cases"].append({"case": f"torchREF-{tag}",
                             "wall_ms": wall(lambda: torch.cumulative_trapezoid(y, dx=1.0, dim=-1))})
    with open(os.path.join(ART, "bench4.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1)[:24000])


if __name__ == "__main__":
    main()
