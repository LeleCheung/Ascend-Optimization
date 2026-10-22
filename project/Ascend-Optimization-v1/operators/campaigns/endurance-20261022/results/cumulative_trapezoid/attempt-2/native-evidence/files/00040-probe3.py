"""Bounded diagnostic: long-row scan formulations (correctness vs torch + walltime)."""
import json
import os
import time
import traceback

import torch
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
DEV = "npu:0"
OUT = {}


def sync():
    torch.npu.synchronize()


def bench(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1e3


@triton.jit
def k_mask2(y_ptr, out_ptr, N: tl.constexpr, C: tl.constexpr, B: tl.constexpr,
            NCH: tl.constexpr):
    row = tl.program_id(0)
    ck = tl.arange(0, C)
    bk = tl.arange(0, B)
    cj = tl.arange(0, C)
    tril_c = cj[None, :] <= ck[:, None]
    tril_b = bk[None, :] <= bk[:, None]
    carry = tl.zeros((1,), dtype=tl.float32)
    yb = y_ptr + row * N
    ob = out_ptr + row * (N - 1)
    for n in range(NCH):
        idx = n * (C * B) + ck[:, None] * B + bk[None, :]
        m = idx < (N - 1)
        a = tl.load(yb + idx, mask=m, other=0.0).to(tl.float32)
        b = tl.load(yb + idx + 1, mask=m, other=0.0).to(tl.float32)
        s = (a + b) * 0.5
        loc = tl.sum(tl.where(tril_b[None, :, :], s[:, None, :], 0.0), axis=2)
        tot = tl.sum(s, axis=1)
        coff = tl.sum(tl.where(tril_c, tot[None, :], 0.0), axis=1)
        cs = loc + (coff - tot)[:, None] + carry
        tl.store(ob + idx, cs, mask=m)
        carry = carry + tl.sum(tl.where((ck == C - 1), coff, 0.0), axis=0)


@triton.jit
def k_cs2d(y_ptr, out_ptr, N: tl.constexpr, C: tl.constexpr, B: tl.constexpr,
           NCH: tl.constexpr):
    row = tl.program_id(0)
    ck = tl.arange(0, C)
    bk = tl.arange(0, B)
    tril_c = bk[None, :] <= ck[:, None]
    carry = tl.zeros((1,), dtype=tl.float32)
    yb = y_ptr + row * N
    ob = out_ptr + row * (N - 1)
    for n in range(NCH):
        idx = n * (C * B) + ck[:, None] * B + bk[None, :]
        m = idx < (N - 1)
        a = tl.load(yb + idx, mask=m, other=0.0).to(tl.float32)
        b = tl.load(yb + idx + 1, mask=m, other=0.0).to(tl.float32)
        s = (a + b) * 0.5
        loc = tl.cumsum(s, axis=1)
        tot = tl.sum(s, axis=1)
        coff = tl.sum(tl.where(tril_c, tot[None, :], 0.0), axis=1)
        cs = loc + (coff - tot)[:, None] + carry
        tl.store(ob + idx, cs, mask=m)
        carry = carry + tl.sum(tl.where((ck == C - 1), coff, 0.0), axis=0)


@triton.jit
def k_rowcs(y_ptr, out_ptr, N: tl.constexpr, NB: tl.constexpr, BR: tl.constexpr):
    """BR rows per program, scan along axis 1 of a (BR, NB) block."""
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    bk = tl.arange(0, NB)
    m = bk < (N - 1)
    addr = rows[:, None] * N + bk[None, :]
    a = tl.load(y_ptr + addr, mask=m[None, :], other=0.0).to(tl.float32)
    b = tl.load(y_ptr + addr + 1, mask=m[None, :], other=0.0).to(tl.float32)
    cs = tl.cumsum((a + b) * 0.5, axis=1)
    tl.store(out_ptr + rows[:, None] * (N - 1) + bk[None, :], cs, mask=m[None, :])


def run(shape, dt, tag):
    y = torch.randn(shape, device=DEV).to(dt)
    rows, N = shape
    ref = torch.cumulative_trapezoid(y, dx=1, dim=-1)
    o = torch.empty((rows, N - 1), dtype=dt, device=DEV)
    e = {"ref_ms": bench(lambda: torch.cumulative_trapezoid(y, dx=1, dim=-1))}
    for (C, B) in ((16, 16), (64, 16), (8, 128), (32, 32)):
        nch = triton.cdiv(N - 1, C * B)

        def go(C=C, B=B, nch=nch):
            k_mask2[(rows,)](y, o, N, C=C, B=B, NCH=nch)

        try:
            go()
            sync()
            e[f"mask2_C{C}xB{B}_err"] = float((o.float() - ref.float()).abs().max())
            e[f"mask2_C{C}xB{B}_ms"] = bench(go)
        except Exception:
            e[f"mask2_C{C}xB{B}_err"] = traceback.format_exc()[-250:]
    for (C, B) in ((16, 32), (64, 16)):
        nch = triton.cdiv(N - 1, C * B)

        def go(C=C, B=B, nch=nch):
            k_cs2d[(rows,)](y, o, N, C=C, B=B, NCH=nch)

        try:
            go()
            sync()
            e[f"cs2d_C{C}xB{B}_err"] = float((o.float() - ref.float()).abs().max())
            e[f"cs2d_C{C}xB{B}_ms"] = bench(go)
        except Exception:
            e[f"cs2d_C{C}xB{B}_err"] = traceback.format_exc()[-250:]
    for (BR, NB) in ((4, 4096), (8, 4096), (2, 4096)):
        g = (triton.cdiv(rows, BR),)

        def go(BR=BR, NB=NB, g=g):
            k_rowcs[g](y, o, N, NB=NB, BR=BR)

        try:
            go()
            sync()
            e[f"rowcs_BR{BR}_err"] = float((o.float() - ref.float()).abs().max())
            e[f"rowcs_BR{BR}_ms"] = bench(go)
        except Exception:
            e[f"rowcs_BR{BR}_err"] = traceback.format_exc()[-250:]
    # gate-shaped check for the winner candidates
    yg = torch.randn((4, 4097), device=DEV).to(torch.float16)
    refg = torch.cumulative_trapezoid(yg, dx=0.125, dim=-1)
    og = torch.empty((4, 4096), dtype=torch.float16, device=DEV)
    for (C, B) in ((64, 16), (16, 16)):
        try:
            k_mask2[(4,)](yg, og, 4097, C=C, B=B, NCH=triton.cdiv(4096, C * B))
            sync()
            e[f"gate_mask2_C{C}xB{B}_err"] = float(
                (og.float() - refg.float()).abs().max())
        except Exception:
            e[f"gate_mask2_C{C}_err"] = traceback.format_exc()[-200:]
    OUT[tag] = e


def main():
    for tag, dt, shape in (
        ("fp16_1024x4096", torch.float16, (1024, 4096)),
        ("fp32_1024x4096", torch.float32, (1024, 4096)),
    ):
        try:
            run(shape, dt, tag)
        except Exception:
            OUT[tag + "_fatal"] = traceback.format_exc()
    with open(os.path.join(ART, "probe3.json"), "w") as f:
        json.dump(OUT, f, indent=1)
    print(json.dumps(OUT, indent=1)[:6000])


main()
