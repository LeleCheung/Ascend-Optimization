"""Probe 3: single-launch fused 2x kernel variants built only from proven-fast
patterns (1D contiguous shifted loads, 2D join, 3D store), boundary clamping
via pointwise arithmetic instead of tl.where, plus host full-call floors.
"""

import json
import os
import time

import torch
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
RES = {}


def rec(k, v):
    RES[k] = v
    try:
        with open(os.path.join(ART, "probe3.json"), "w") as f:
            json.dump(RES, f, indent=1, default=str)
    except Exception as exc:
        print("artifact io error", exc, flush=True)
    print("RESULT", k, v, flush=True)


@triton.jit
def k_fused9(o, i, PAD: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
             ROWS: tl.constexpr, NBLK: tl.constexpr, ARITH: tl.constexpr):
    """Fused 2x bilinear (align_corners=False, no explicit scales).

    9 shifted contiguous loads; row/column edge clamps by pointwise arithmetic
    (or uniform control flow when ARITH is False).
    """
    pid = tl.program_id(0)
    img = pid // NBLK
    blk = pid % NBLK
    b0 = PAD + img * (IH * IW) + blk * (ROWS * IW)
    off = tl.arange(0, ROWS * IW)
    kk = off % IW

    a0 = tl.load(i + b0 - IW + off).to(tl.float32)
    b0v = tl.load(i + b0 + off).to(tl.float32)
    c0 = tl.load(i + b0 + IW + off).to(tl.float32)

    ve0 = 0.25 * a0 + 0.75 * b0v
    vd0 = 0.75 * b0v + 0.25 * c0
    am = tl.load(i + b0 - IW - 1 + off).to(tl.float32)
    bm = tl.load(i + b0 - 1 + off).to(tl.float32)
    cm = tl.load(i + b0 + IW - 1 + off).to(tl.float32)
    ap = tl.load(i + b0 - IW + 1 + off).to(tl.float32)
    bp = tl.load(i + b0 + 1 + off).to(tl.float32)
    cp = tl.load(i + b0 + IW + 1 + off).to(tl.float32)
    vem = 0.25 * am + 0.75 * bm
    vep = 0.25 * ap + 0.75 * bp
    vdm = 0.75 * bm + 0.25 * cm
    vdp = 0.75 * bp + 0.25 * cp

    c0f = (kk == 0).to(tl.float32)
    c1f = (kk == IW - 1).to(tl.float32)
    vem = vem + (ve0 - vem) * c0f
    vdm = vdm + (vd0 - vdm) * c0f
    vep = vep + (ve0 - vep) * c1f
    vdp = vdp + (vd0 - vdp) * c1f

    firstm = ((blk == 0) & (off < IW)).to(tl.float32)
    lastm = ((blk == NBLK - 1) & (off >= (ROWS - 1) * IW)).to(tl.float32)
    if ARITH:
        ve0 = ve0 + (b0v - ve0) * firstm
        vem = vem + (bm - vem) * firstm
        vep = vep + (bp - vep) * firstm
        vd0 = vd0 + (b0v - vd0) * lastm
        vdm = vdm + (bm - vdm) * lastm
        vdp = vdp + (bp - vdp) * lastm
    else:
        if blk == 0:
            ve0 = ve0 + (b0v - ve0) * firstm
            vem = vem + (bm - vem) * firstm
            vep = vep + (bp - vep) * firstm
        if blk == NBLK - 1:
            vd0 = vd0 + (b0v - vd0) * lastm
            vdm = vdm + (bm - vdm) * lastm
            vdp = vdp + (bp - vdp) * lastm

    he = tl.reshape(tl.join(tl.reshape(vem, (ROWS, IW)),
                            tl.reshape(ve0, (ROWS, IW))), (ROWS, IW, 2))
    hd = tl.reshape(tl.join(tl.reshape(vdm, (ROWS, IW)),
                            tl.reshape(vd0, (ROWS, IW))), (ROWS, IW, 2))

    rows = blk * ROWS + tl.arange(0, ROWS)
    j = tl.arange(0, 2)
    offs = (img * (2 * IH * 2 * IW) + (rows * 2)[:, None, None] * (2 * IW)
            + tl.arange(0, IW)[None, :, None] * 2 + j[None, None, :])
    tl.store(o + offs, he.to(o.dtype.element_ty))
    tl.store(o + offs + 2 * IW, hd.to(o.dtype.element_ty))


@triton.jit
def k_fused6(o, i, PAD: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
             ROWS: tl.constexpr, NBLK: tl.constexpr):
    """6-load variant: pair (odd[k-1], even[k]) covers columns 2k-1..2k,
    stored at base 2k-1 with the single row-start element masked out.
    NOTE: column OW-1 of every row is not produced by this variant."""
    pid = tl.program_id(0)
    img = pid // NBLK
    blk = pid % NBLK
    b0 = PAD + img * (IH * IW) + blk * (ROWS * IW)
    off = tl.arange(0, ROWS * IW)
    kk = off % IW
    c0f = (kk == 0).to(tl.float32)

    a0 = tl.load(i + b0 - IW + off).to(tl.float32)
    b0v = tl.load(i + b0 + off).to(tl.float32)
    c0 = tl.load(i + b0 + IW + off).to(tl.float32)
    am = tl.load(i + b0 - IW - 1 + off).to(tl.float32)
    bm = tl.load(i + b0 - 1 + off).to(tl.float32)
    cm = tl.load(i + b0 + IW - 1 + off).to(tl.float32)

    ve0 = 0.25 * a0 + 0.75 * b0v
    vd0 = 0.75 * b0v + 0.25 * c0
    vem = 0.25 * am + 0.75 * bm
    vdm = 0.75 * bm + 0.25 * cm

    firstm = ((blk == 0) & (off < IW)).to(tl.float32)
    lastm = ((blk == NBLK - 1) & (off >= (ROWS - 1) * IW)).to(tl.float32)
    ve0 = ve0 + (b0v - ve0) * firstm
    vem = vem + (bm - vem) * firstm
    vd0 = vd0 + (b0v - vd0) * lastm
    vdm = vdm + (bm - vdm) * lastm
    # column clamp after the row clamp so the row-start corner is exact
    vem = vem + (ve0 - vem) * c0f
    vdm = vdm + (vd0 - vdm) * c0f

    he_o = 0.75 * vem + 0.25 * ve0
    he_e = 0.25 * vem + 0.75 * ve0
    hd_o = 0.75 * vdm + 0.25 * vd0
    hd_e = 0.25 * vdm + 0.75 * vd0

    he = tl.reshape(tl.join(tl.reshape(he_o, (ROWS, IW)),
                            tl.reshape(he_e, (ROWS, IW))), (ROWS, IW, 2))
    hd = tl.reshape(tl.join(tl.reshape(hd_o, (ROWS, IW)),
                            tl.reshape(hd_e, (ROWS, IW))), (ROWS, IW, 2))

    rows = blk * ROWS + tl.arange(0, ROWS)
    j = tl.arange(0, 2)
    offs = (img * (2 * IH * 2 * IW) + (rows * 2)[:, None, None] * (2 * IW)
            + tl.arange(0, IW)[None, :, None] * 2 - 1 + j[None, None, :])
    m = (tl.arange(0, IW)[None, :, None] > 0) | (j[None, None, :] > 0)
    mm = tl.broadcast_to(m, (ROWS, IW, 2))
    tl.store(o + offs, he.to(o.dtype.element_ty), mask=mm)
    tl.store(o + offs + 2 * IW, hd.to(o.dtype.element_ty), mask=mm)


def dev_ms(fn, K=50, reps=3):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    best = float("inf")
    for _ in range(reps):
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        for _ in range(K):
            fn()
        e.record()
        torch.npu.synchronize()
        best = min(best, s.elapsed_time(e) / K)
    return best


def wall_ms(fn, K=200):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(K):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / K * 1e3


def cmp_report(got, ref, OW):
    d = (got - ref).abs()
    idx = int(d.argmax())
    col = idx % OW
    bad = int((d > 0.002).sum())
    d2 = d.clone()
    d2[(torch.arange(d.numel(), device=d.device) % OW) == (OW - 1)] = 0.0
    return {"maxdiff": float(d.max()), "argmax_col": col, "n_bad": bad,
            "maxdiff_excl_lastcol": float(d2.max())}


@triton.jit
def k_two_c(o, i, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(o + off, tl.load(i + off))


def main():
    dev = torch.device("npu:0")
    torch.npu.set_device(dev)
    NC, IH, IW = 128, 128, 128
    OH, OW = 2 * IH, 2 * IW
    PAD = IW
    n_in = NC * IH * IW
    n_out = NC * OH * OW

    for dt in (torch.float32, torch.float16):
        tag = "f32" if dt is torch.float32 else "f16"
        ibuf = torch.randn(n_in + 2 * PAD, device=dev, dtype=dt)
        obuf = torch.empty(n_out, device=dev, dtype=dt)
        x = ibuf[PAD:PAD + n_in].view(1, NC, IH, IW)
        ref = torch.nn.functional.interpolate(
            x, size=(OH, OW), mode="bilinear", align_corners=False).reshape(-1)

        for ROWS in (8, 16):
            nblk = IH // ROWS
            grid = (NC * nblk,)
            for arith in (True, False):
                key = f"f9_r{ROWS}_{'arith' if arith else 'if'}_{tag}"
                try:
                    f = lambda ROWS=ROWS, nblk=nblk, grid=grid, arith=arith: k_fused9[grid](
                        obuf, ibuf, PAD=PAD, IH=IH, IW=IW, ROWS=ROWS, NBLK=nblk,
                        ARITH=arith)
                    ms = dev_ms(f)
                    obuf.zero_()
                    f()
                    torch.npu.synchronize()
                    out = {"ms": ms}
                    out.update(cmp_report(obuf, ref, OW))
                    rec(key, out)
                except Exception as exc:
                    rec(key, {"error": str(exc)[:300]})
            try:
                f = lambda ROWS=ROWS, nblk=nblk, grid=grid: k_fused6[grid](
                    obuf, ibuf, PAD=PAD, IH=IH, IW=IW, ROWS=ROWS, NBLK=nblk)
                ms = dev_ms(f)
                obuf.zero_()
                f()
                torch.npu.synchronize()
                out = {"ms": ms}
                out.update(cmp_report(obuf, ref, OW))
                rec(f"f6_r{ROWS}_{tag}", out)
            except Exception as exc:
                rec(f"f6_r{ROWS}_{tag}", {"error": str(exc)[:300]})
        del ibuf, obuf

    # ---- small geometry device time (262144 outputs) ----
    NC2, IH2, IW2 = 16, 64, 64
    x = torch.randn(1, NC2, IH2, IW2, device=dev, dtype=torch.float16)
    ref2 = torch.nn.functional.interpolate(
        x, size=(2 * IH2, 2 * IW2), mode="bilinear",
        align_corners=False).reshape(-1)
    ibuf = torch.zeros(NC2 * IH2 * IW2 + 2 * IW2, device=dev, dtype=torch.float16)
    ibuf[IW2:IW2 + NC2 * IH2 * IW2] = x.reshape(-1)
    obuf = torch.empty(NC2 * 4 * IH2 * IW2, device=dev, dtype=torch.float16)
    for ROWS in (4, 8, 16):
        nblk = IH2 // ROWS
        g = (NC2 * nblk,)
        f = lambda ROWS=ROWS, nblk=nblk, g=g: k_fused9[g](
            obuf, ibuf, PAD=IW2, IH=IH2, IW=IW2, ROWS=ROWS, NBLK=nblk, ARITH=True)
        try:
            ms = dev_ms(f)
            obuf.zero_()
            f()
            torch.npu.synchronize()
            out = {"ms": ms}
            out.update(cmp_report(obuf, ref2, 2 * IW2))
            rec(f"small_f9_r{ROWS}", out)
        except Exception as exc:
            rec(f"small_f9_r{ROWS}", {"error": str(exc)[:300]})

    # ---- host full-call floors ----
    ii = torch.empty(8192, device=dev, dtype=torch.float16)

    def call_1c():
        o = torch.empty(8192, device=dev, dtype=torch.float16)
        k_two_c[(1,)](o, ii, BLOCK=8192)
        return o

    rec("fullcall_1constexpr_dev", dev_ms(call_1c))
    rec("fullcall_1constexpr_wall", wall_ms(call_1c))

    nblk = IH2 // 8
    g = (NC2 * nblk,)

    def call_f9():
        o = torch.empty(NC2 * 4 * IH2 * IW2, device=dev, dtype=torch.float16)
        k_fused9[g](o, ibuf, PAD=IW2, IH=IH2, IW=IW2, ROWS=8, NBLK=nblk, ARITH=True)
        return o

    rec("fullcall_f9_dev", dev_ms(call_f9))
    rec("fullcall_f9_wall", wall_ms(call_f9))
    rec("done", True)


if __name__ == "__main__":
    main()
