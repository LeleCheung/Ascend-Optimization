"""Bounded diagnostic: compare non-tl.cumsum scan schemes for long contiguous rows.

Variant A: masked-broadcast intra-tile scan (Vector only, dtype agnostic).
Variant B: tril-matrix tl.dot intra-tile scan over R rows (Cube), serial carry.
Diagnostic evidence only.
"""
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
def k_mask(y_ptr, out_ptr, N: tl.constexpr, B: tl.constexpr, NCH: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, B)
    j = tl.arange(0, B)
    tril = j[None, :] <= k[:, None]
    last = k == (B - 1)
    carry = tl.zeros((1,), dtype=tl.float32)
    yb = y_ptr + row * N
    ob = out_ptr + row * (N - 1)
    for c in range(NCH):
        idx = c * B + k
        m = idx < (N - 1)
        a = tl.load(yb + idx, mask=m, other=0.0).to(tl.float32)
        b = tl.load(yb + idx + 1, mask=m, other=0.0).to(tl.float32)
        s = (a + b) * 0.5
        cs = tl.sum(tl.where(tril, s[None, :], 0.0), axis=1) + carry
        tl.store(ob + idx, cs, mask=m)
        carry = tl.sum(tl.where(last, cs, 0.0), axis=0)
    carry = carry


@triton.jit
def k_mask_2lvl(y_ptr, out_ptr, N: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                NCH: tl.constexpr):
    """Tile of B*C elements: intra-row (fast axis) masked scan + masked scan of
    C row sums (both vector only)."""
    row = tl.program_id(0)
    yb = y_ptr + row * N
    ob = out_ptr + row * (N - 1)
    ck = tl.arange(0, C)
    bk = tl.arange(0, B)
    cj = tl.arange(0, C)
    tril_c = cj[None, :] <= ck[:, None]
    last_c = ck == (C - 1)
    carry = tl.zeros((1,), dtype=tl.float32)
    for n in range(NCH):
        base = n * (B * C)
        idx = base + ck[:, None] * B + bk[None, :]  # (C,B) flat order
        m = idx < (N - 1)
        a = tl.load(yb + idx, mask=m, other=0.0).to(tl.float32)
        b = tl.load(yb + idx + 1, mask=m, other=0.0).to(tl.float32)
        s = (a + b) * 0.5
        # scan along fast axis B via prefix-sum of s over bk (log via broadcasting)
        loc = tl.sum(tl.where(bk[None, :, None] >= bk[None, None, :],
                              s[:, None, :], 0.0), axis=2)
        tot = tl.sum(s, axis=1)
        coff = tl.sum(tl.where(tril_c, tot[None, :], 0.0), axis=1)
        tot_all = tl.sum(tl.where(last_c, coff, 0.0), axis=0)
        cs = loc + (coff - tot)[:, None] + carry
        tl.store(ob + idx, cs, mask=m)
        carry = carry + tot_all
    carry = carry


@triton.jit
def k_dot(y_ptr, out_ptr, N: tl.constexpr, B: tl.constexpr, R: tl.constexpr,
          NCH: tl.constexpr):
    rb = tl.program_id(0)
    rows = rb * R + tl.arange(0, R)
    kk = tl.arange(0, B)
    jj = tl.arange(0, B)
    tril = tl.where(jj[None, :] <= kk[:, None], 1.0, 0.0).to(tl.float16)
    carry = tl.zeros((B, R), dtype=tl.float32)
    base = rows[None, :] * N
    obase = rows[None, :] * (N - 1)
    for c in range(NCH):
        idx = c * B + kk
        m = (idx < (N - 1))[:, None]
        addr = base + idx[:, None]
        a = tl.load(y_ptr + addr, mask=m, other=0.0).to(tl.float32)
        b = tl.load(y_ptr + addr + 1, mask=m, other=0.0).to(tl.float32)
        s = ((a + b) * 0.5).to(tl.float16)
        cs = tl.dot(tril, s) + carry
        tl.store(out_ptr + obase + idx[:, None], cs, mask=m)
        carry = carry + tl.sum(tl.where((kk == B - 1)[:, None], cs, 0.0), axis=0)[None, :]
    carry = carry


def run_variant(e, name, launch, y, ref, dt, shape):
    try:
        launch()
        sync()
        err = None
        e[name + "_err"] = err
        e[name + "_ms"] = bench(launch)
    except Exception:
        e[name + "_err"] = traceback.format_exc()[-400:]


def part(shape, dt, tag):
    y = torch.randn(shape, device=DEV).to(dt)
    rows, N = shape
    ref = torch.cumulative_trapezoid(y, dx=1, dim=-1)
    o = torch.empty((rows, N - 1), dtype=dt, device=DEV)
    e = {"ref_ms": bench(lambda: torch.cumulative_trapezoid(y, dx=1, dim=-1))}
    for B in (16, 32, 64, 128):
        nch = triton.cdiv(N - 1, B)

        def go(B=B, nch=nch):
            k_mask[(rows,)](y, o, N, B=B, NCH=nch)

        try:
            go()
            sync()
            e[f"mask_B{B}_err"] = float((o.float() - ref.float()).abs().max())
            e[f"mask_B{B}_ms"] = bench(go)
        except Exception:
            e[f"mask_B{B}_err"] = traceback.format_exc()[-300:]
    for (B, C) in ((32, 32), (64, 16), (16, 64)):
        n = B * C
        nch = triton.cdiv(N - 1, n)

        def go(B=B, C=C, nch=nch):
            k_mask_2lvl[(rows,)](y, o, N, B=B, C=C, NCH=nch)

        try:
            go()
            sync()
            e[f"mask2_B{B}x{C}_err"] = float((o.float() - ref.float()).abs().max())
            e[f"mask2_B{B}x{C}_ms"] = bench(go)
        except Exception:
            e[f"mask2_B{B}x{C}_err"] = traceback.format_exc()[-300:]
    for (B, R) in ((64, 32), (128, 16), (128, 32)):
        nch = triton.cdiv(N - 1, B)
        g = (triton.cdiv(rows, R),)

        def go(B=B, R=R, nch=nch, g=g):
            k_dot[g](y, o, N, B=B, R=R, NCH=nch)

        try:
            go()
            sync()
            e[f"dot_B{B}xR{R}_err"] = float((o.float() - ref.float()).abs().max())
            e[f"dot_B{B}R{R}_ms"] = bench(go)
        except Exception:
            e[f"dot_B{B}R{R}_err"] = traceback.format_exc()[-300:]
    OUT[tag] = e


def main():
    for tag, dt, shape in (
        ("fp16_1024x4096", torch.float16, (1024, 4096)),
        ("fp32_1024x4096", torch.float32, (1024, 4096)),
        ("fp16_128x1024", torch.float16, (128, 1024)),
    ):
        try:
            part(shape, dt, tag)
        except Exception:
            OUT[tag + "_fatal"] = traceback.format_exc()
    with open(os.path.join(ART, "probe2.json"), "w") as f:
        json.dump(OUT, f, indent=1)
    print(json.dumps(OUT, indent=1)[:8000])


main()
