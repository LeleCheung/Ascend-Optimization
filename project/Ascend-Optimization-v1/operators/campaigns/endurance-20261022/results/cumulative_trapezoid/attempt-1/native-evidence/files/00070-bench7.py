"""Isolate why the bench5 dot-prefix kernel was fast (0.16ms) and bench6's was
slow (11ms) on (1024,4096): loop-carried tile vs vector, masked vs unmasked.
"""
import json
import os
import time

import torch
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
res = {"cases": [], "errors": []}


# A: bench5 formulation, loop-carried [BR,L] tile carry, unmasked (buggy store)
@triton.jit
def kA(y, o, NOUT: tl.constexpr, N: tl.constexpr, BR: tl.constexpr, L: tl.constexpr,
       SCALE):
    pr = tl.program_id(0)
    r = pr * BR + tl.arange(0, BR)
    c = tl.arange(0, L)
    seg = (c[:, None] >= c[None, :]).to(tl.float16)
    carry = tl.zeros((BR, L), dtype=tl.float32)
    for cc in range(0, N // L):
        kk = cc * L + c
        a = tl.load(y + r[:, None] * N + kk[None, :]).to(tl.float16)
        b = tl.load(y + r[:, None] * N + (kk + 1)[None, :]).to(tl.float16)
        t = ((a + b) * SCALE).to(tl.float16)
        lp = tl.dot(t, seg).to(tl.float32) + carry
        tl.store(o + r[:, None] * NOUT + kk[None, :], lp)
        last = tl.sum(tl.where(c[None, :] == (L - 1), lp, 0.0), axis=1)
        carry = last[:, None] + tl.zeros((BR, L), dtype=tl.float32)


# A2: same, but correct: loop bound covers NOUT and the store is masked
@triton.jit
def kA2(y, o, NOUT: tl.constexpr, N: tl.constexpr, BR: tl.constexpr, L: tl.constexpr,
        SCALE):
    pr = tl.program_id(0)
    r = pr * BR + tl.arange(0, BR)
    c = tl.arange(0, L)
    seg = (c[:, None] >= c[None, :]).to(tl.float16)
    carry = tl.zeros((BR, L), dtype=tl.float32)
    for cc in range(0, (NOUT + L - 1) // L):
        kk = cc * L + c
        m = (kk < NOUT)[None, :]
        a = tl.load(y + r[:, None] * N + kk[None, :], mask=m, other=0.0).to(tl.float16)
        b = tl.load(y + r[:, None] * N + (kk + 1)[None, :], mask=m, other=0.0).to(tl.float16)
        t = ((a + b) * SCALE).to(tl.float16)
        lp = tl.dot(t, seg).to(tl.float32) + carry
        tl.store(o + r[:, None] * NOUT + kk[None, :], lp, mask=m)
        last = tl.sum(tl.where(c[None, :] == (L - 1), lp, 0.0), axis=1)
        carry = last[:, None] + tl.zeros((BR, L), dtype=tl.float32)


# A3: tile carry, store mask, unmasked loads with clamped indices
@triton.jit
def kA3(y, o, NOUT: tl.constexpr, N: tl.constexpr, BR: tl.constexpr, L: tl.constexpr,
        SCALE):
    pr = tl.program_id(0)
    r = pr * BR + tl.arange(0, BR)
    c = tl.arange(0, L)
    seg = (c[:, None] >= c[None, :]).to(tl.float16)
    carry = tl.zeros((BR, L), dtype=tl.float32)
    for cc in range(0, (NOUT + L - 1) // L):
        kk = cc * L + c
        kc = tl.minimum(kk, NOUT - 1)
        a = tl.load(y + r[:, None] * N + kc[None, :]).to(tl.float16)
        b = tl.load(y + r[:, None] * N + (kc + 1)[None, :]).to(tl.float16)
        t = ((a + b) * SCALE).to(tl.float16)
        lp = tl.dot(t, seg).to(tl.float32) + carry
        tl.store(o + r[:, None] * NOUT + kk[None, :], lp, mask=(kk < NOUT)[None, :])
        last = tl.sum(tl.where(c[None, :] == (L - 1), lp, 0.0), axis=1)
        carry = last[:, None] + tl.zeros((BR, L), dtype=tl.float32)


def wall(fn, iters=20, warm=6):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def add(name, fn, out, ref, atol):
    try:
        fn()
        torch.npu.synchronize()
        d = (out.float() - ref.float()).abs()
        bad = (d > atol + atol * ref.float().abs()).sum().item()
        res["cases"].append({"case": name, "wall_ms": wall(fn), "maxabs": d.max().item(),
                             "bad": bad, "n": ref.numel()})
    except Exception as e:
        res["errors"].append({"case": name, "err": repr(e)[-220:]})


def main():
    torch.npu.set_device(0)
    for R, N, dt in ((1024, 4096, "float16"), (128, 1024, "float16"), (1024, 4096, "float32")):
        tdt = getattr(torch, dt)
        y = torch.randn(R, N, generator=torch.Generator().manual_seed(1730)).to(tdt).npu()
        ref = torch.cumulative_trapezoid(y, dx=1.0, dim=-1)
        atol = 0.015 if dt != "float32" else 1e-4
        res["cases"].append({"case": f"torchREF-{dt}-{R}x{N}",
                             "wall_ms": wall(lambda: torch.cumulative_trapezoid(y, dx=1.0, dim=-1))})
        for kern, tag in ((kA, "A"), (kA2, "A2"), (kA3, "A3")):
            for BR, L in ((16, 128), (32, 128), (32, 64)):
                out = torch.empty(R, N - 1, dtype=tdt, device="npu:0")
                g = (triton.cdiv(R, BR),)
                name = f"{tag}-{dt}-{R}x{N}-br{BR}-L{L}"
                add(name, lambda kern=kern, g=g, BR=BR, L=L, out=out, y=y, R=R, N=N:
                    kern[g](y, out, NOUT=N - 1, N=N, BR=BR, L=L, SCALE=0.5), out, ref, atol)
    with open(os.path.join(ART, "bench7.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1)[:20000])


if __name__ == "__main__":
    main()
