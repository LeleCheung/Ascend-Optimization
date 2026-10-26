"""Probe 2: candidate architectures + host launch decomposition.

A. device: fused 9-load (shifted contiguous loads, no gather, exact) vs a
   separable 2-kernel (vertical row-expand, then horizontal column-expand),
   with exact numeric verification against torch for both.
B. host: marginal cost of a Triton launch (no args / 2 ptr args / +constexpr),
   plain JITFunction vs LibEntry, torch.empty, single small-shape full call.
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
        with open(os.path.join(ART, "probe2.json"), "w") as f:
            json.dump(RES, f, indent=1, default=str)
    except Exception as exc:
        print("artifact io error", exc, flush=True)
    print("RESULT", k, v, flush=True)


# ---------------------------------------------------------------- kernels
@triton.jit
def k_two(o, i):
    off = tl.arange(0, 8192)
    tl.store(o + off, tl.load(i + off))


@triton.jit
def k_two_c(o, i, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(o + off, tl.load(i + off))


@triton.jit
def k_f9(o, i, PAD: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
         ROWS: tl.constexpr, NBLK: tl.constexpr):
    """Single fused kernel: 9 shifted contiguous loads, no gather, exact."""
    pid = tl.program_id(0)
    img = pid // NBLK
    blk = pid % NBLK
    b0 = PAD + img * (IH * IW) + blk * (ROWS * IW)
    off = tl.arange(0, ROWS * IW)
    kk = off % IW
    e0 = kk == 0
    e1 = kk == IW - 1
    first = (blk == 0) & (off < IW)
    last = (blk == NBLK - 1) & (off >= (ROWS - 1) * IW)

    a0 = tl.load(i + b0 - IW + off).to(tl.float32)
    b0v = tl.load(i + b0 + off).to(tl.float32)
    c0 = tl.load(i + b0 + IW + off).to(tl.float32)
    a0 = tl.where(first, b0v, a0)
    c0 = tl.where(last, b0v, c0)

    am = tl.where(e0, a0, tl.load(i + b0 - IW - 1 + off).to(tl.float32))
    bm = tl.where(e0, b0v, tl.load(i + b0 - 1 + off).to(tl.float32))
    cm = tl.where(e0, c0, tl.load(i + b0 + IW - 1 + off).to(tl.float32))
    ap = tl.where(e1, a0, tl.load(i + b0 - IW + 1 + off).to(tl.float32))
    bp = tl.where(e1, b0v, tl.load(i + b0 + 1 + off).to(tl.float32))
    cp = tl.where(e1, c0, tl.load(i + b0 + IW + 1 + off).to(tl.float32))
    am = tl.where(first, bm, am)
    ap = tl.where(first, bp, ap)
    cm = tl.where(last, bm, cm)
    cp = tl.where(last, bp, cp)

    ve0 = 0.25 * a0 + 0.75 * b0v
    vem = 0.25 * am + 0.75 * bm
    vep = 0.25 * ap + 0.75 * bp
    vd0 = 0.75 * b0v + 0.25 * c0
    vdm = 0.75 * bm + 0.25 * cm
    vdp = 0.75 * bp + 0.25 * cp

    he = tl.join(0.25 * vem + 0.75 * ve0, 0.75 * ve0 + 0.25 * vep)
    hd = tl.join(0.25 * vdm + 0.75 * vd0, 0.75 * vd0 + 0.25 * vdp)
    he = tl.reshape(he, (ROWS, IW, 2))
    hd = tl.reshape(hd, (ROWS, IW, 2))

    rows = blk * ROWS + tl.arange(0, ROWS)
    ob = img * (2 * IH * 2 * IW)
    j = tl.arange(0, 2)
    offs = (ob + (rows * 2)[:, None, None] * (2 * IW)
            + tl.arange(0, IW)[None, :, None] * 2 + j[None, None, :])
    tl.store(o + offs, he.to(o.dtype.element_ty))
    tl.store(o + offs + 2 * IW, hd.to(o.dtype.element_ty))


@triton.jit
def k_vrow(o, i, PAD: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
           ROWS: tl.constexpr, NBLK: tl.constexpr):
    """Vertical pass: 2x rows, compact contiguous intermediate (2*IH, IW)."""
    pid = tl.program_id(0)
    img = pid // NBLK
    blk = pid % NBLK
    b0 = PAD + img * (IH * IW) + blk * (ROWS * IW)
    off = tl.arange(0, ROWS * IW)
    first = (blk == 0) & (off < IW)
    last = (blk == NBLK - 1) & (off >= (ROWS - 1) * IW)
    a = tl.load(i + b0 - IW + off).to(tl.float32)
    b = tl.load(i + b0 + off).to(tl.float32)
    c = tl.load(i + b0 + IW + off).to(tl.float32)
    a = tl.where(first, b, a)
    c = tl.where(last, b, c)
    e = 0.25 * a + 0.75 * b
    d = 0.75 * b + 0.25 * c
    rows = blk * ROWS + tl.arange(0, ROWS)
    ob = 1 + img * (2 * IH * IW) + (rows * 2)[:, None] * IW + tl.arange(0, IW)[None, :]
    tl.store(o + ob, tl.reshape(e, (ROWS, IW)).to(o.dtype.element_ty))
    tl.store(o + ob + IW, tl.reshape(d, (ROWS, IW)).to(o.dtype.element_ty))


@triton.jit
def k_hcol(o, h, IH: tl.constexpr, IW: tl.constexpr, R2: tl.constexpr):
    """Horizontal pass: 2x columns from a contiguous (rows, IW) intermediate."""
    pid = tl.program_id(0)
    b0 = 1 + pid * R2 * IW
    off = tl.arange(0, R2 * IW)
    kk = off % IW
    v = tl.load(h + b0 + off).to(tl.float32)
    lm = tl.where(kk == 0, v, tl.load(h + b0 - 1 + off).to(tl.float32))
    rp = tl.where(kk == IW - 1, v, tl.load(h + b0 + 1 + off).to(tl.float32))
    he = tl.join(0.25 * lm + 0.75 * v, 0.75 * v + 0.25 * rp)
    he = tl.reshape(he, (R2, IW, 2))
    rows = pid * R2 + tl.arange(0, R2)
    j = tl.arange(0, 2)
    offs = rows[:, None, None] * (2 * IW) + tl.arange(0, IW)[None, :, None] * 2 + j[None, None, :]
    tl.store(o + offs, he.to(o.dtype.element_ty))


# ---------------------------------------------------------------- harness
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


def wall_ms(fn, K=200, sync_each=False):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    if sync_each:
        for _ in range(K):
            fn()
            torch.npu.synchronize()
    else:
        for _ in range(K):
            fn()
        torch.npu.synchronize()
    return (time.perf_counter() - t0) / K * 1e3


def device_part():
    dev = torch.device("npu:0")
    NC, IH, IW = 128, 128, 128
    OH, OW = 2 * IH, 2 * IW
    PAD = IW
    n_in = NC * IH * IW
    n_out = NC * OH * OW

    for dt in (torch.float32, torch.float16):
        tag = "f32" if dt is torch.float32 else "f16"
        ibuf = torch.randn(n_in + 2 * PAD, device=dev, dtype=dt)
        obuf = torch.empty(n_out, device=dev, dtype=dt)
        hbuf = torch.empty(NC * OH * IW + 2, device=dev, dtype=dt)
        x = ibuf[PAD:PAD + n_in].view(1, NC, IH, IW)
        ref = torch.nn.functional.interpolate(
            x, size=(OH, OW), mode="bilinear", align_corners=False).reshape(-1)

        for ROWS in (4, 8, 16):
            nblk = IH // ROWS
            grid = (NC * nblk,)
            try:
                f = lambda ROWS=ROWS, nblk=nblk, grid=grid: k_f9[grid](
                    obuf, ibuf, PAD=PAD, IH=IH, IW=IW, ROWS=ROWS, NBLK=nblk)
                ms = dev_ms(f)
                f()
                torch.npu.synchronize()
                err = float((obuf - ref).abs().max())
                rec(f"f9_r{ROWS}_{tag}", {"ms": ms, "maxdiff": err})
            except Exception as exc:
                rec(f"f9_r{ROWS}_{tag}", {"error": str(exc)[:400]})

        # two-kernel separable
        for ROWS in (8, 16, 32):
            nblk = IH // ROWS
            gv = (NC * nblk,)
            fv = lambda ROWS=ROWS, nblk=nblk, gv=gv: k_vrow[gv](
                hbuf, ibuf, PAD=PAD, IH=IH, IW=IW, ROWS=ROWS, NBLK=nblk)
            try:
                msv = dev_ms(fv)
                rec(f"vrow_r{ROWS}_{tag}_ms", msv)
            except Exception as exc:
                rec(f"vrow_r{ROWS}_{tag}_ms", {"error": str(exc)[:400]})
                continue
            for R2 in (16, 32):
                nrow = NC * OH
                if nrow % R2:
                    continue
                gh = (nrow // R2,)
                fh = lambda R2=R2, gh=gh: k_hcol[gh](obuf, hbuf, IH=IH, IW=IW, R2=R2)
                try:
                    msh = dev_ms(fh)
                    tot = dev_ms(lambda fv=fv, fh=fh: (fv(), fh()))
                    fv()
                    fh()
                    torch.npu.synchronize()
                    err = float((obuf - ref).abs().max())
                    rec(f"sep_r{ROWS}_h{R2}_{tag}",
                        {"v_ms": msv, "h_ms": msh, "both_ms": tot, "maxdiff": err})
                except Exception as exc:
                    rec(f"sep_r{ROWS}_h{R2}_{tag}", {"error": str(exc)[:400]})
        del ibuf, obuf, hbuf


def host_part():
    dev = torch.device("npu:0")
    oo = torch.empty(8192, device=dev, dtype=torch.float16)
    ii = torch.empty(8192, device=dev, dtype=torch.float16)
    rec("h_noarg_dev", "skipped")
    rec("h_two_dev", dev_ms(lambda: k_two[(1,)](oo, ii)))
    rec("h_two_c_dev", dev_ms(lambda: k_two_c[(1,)](oo, ii, BLOCK=8192)))
    rec("h_two_c_wall", wall_ms(lambda: k_two_c[(1,)](oo, ii, BLOCK=8192)))
    rec("h_empty_small_wall",
        wall_ms(lambda: torch.empty((1, 16, 128, 128), device=dev, dtype=torch.float16)))
    rec("h_empty_large_wall",
        wall_ms(lambda: torch.empty((4, 32, 256, 256), device=dev, dtype=torch.float32)))
    # 2 launches back to back: marginal cost of the second launch
    rec("h_two_launches_dev", dev_ms(lambda: (k_two_c[(1,)](oo, ii, BLOCK=8192),
                                               k_two_c[(1,)](oo, ii, BLOCK=8192))))
    # small-shape fused kernel full device time (262144 outputs)
    xs = torch.randn(1, 16, 64, 64, device=dev, dtype=torch.float16)
    osmall = torch.empty(1, 16, 128, 128, device=dev, dtype=torch.float16)
    NC, IH, IW = 16, 64, 64
    ibuf = torch.zeros(NC * IH * IW + 2 * IW, device=dev, dtype=torch.float16)
    ibuf[IW:IW + NC * IH * IW] = xs.reshape(-1)
    obuf = torch.empty(NC * 4 * IH * IW, device=dev, dtype=torch.float16)
    for ROWS in (4, 8, 16):
        nblk = IH // ROWS
        g = (NC * nblk,)
        f = lambda ROWS=ROWS, nblk=nblk, g=g: k_f9[g](
            obuf, ibuf, PAD=IW, IH=IH, IW=IW, ROWS=ROWS, NBLK=nblk)
        try:
            rec(f"small_f9_r{ROWS}_dev", dev_ms(f))
        except Exception as exc:
            rec(f"small_f9_r{ROWS}_dev", str(exc)[:200])
    # projected full-call cost of the new single-kernel small-shape path:
    # torch.empty + one plain @triton.jit launch (no LibEntry)
    for ROWS in (4, 8):
        nblk = IH // ROWS
        g = (NC * nblk,)

        def call(g=g, ROWS=ROWS, nblk=nblk):
            o = torch.empty(NC * 4 * IH * IW, device=dev, dtype=torch.float16)
            k_f9[g](o, ibuf, PAD=IW, IH=IH, IW=IW, ROWS=ROWS, NBLK=nblk)
            return o

        rec(f"small_fullcall_r{ROWS}_dev", dev_ms(call))
        rec(f"small_fullcall_r{ROWS}_wall", wall_ms(call))

    nrow = NC * 2 * IH
    for R2 in (8, 16, 32, 64):
        g = (nrow // R2,)
        hh = torch.empty(NC * 2 * IH * IW + 2, device=dev, dtype=torch.float16)
        fv = lambda: k_vrow[(NC * IH // 8,)](hh, ibuf, PAD=IW, IH=IH, IW=IW, ROWS=8,
                                             NBLK=IH // 8)
        fh = lambda R2=R2, g=g: k_hcol[g](obuf, hh, IH=IH, IW=IW, R2=R2)
        try:
            rec(f"small_sep_h{R2}_dev", dev_ms(lambda fv=fv, fh=fh: (fv(), fh())))
        except Exception as exc:
            rec(f"small_sep_h{R2}_dev", str(exc)[:200])


def main():
    torch.npu.set_device(torch.device("npu:0"))
    rec("torch", torch.__version__)
    try:
        device_part()
    except Exception:
        import traceback
        rec("device_error", traceback.format_exc())
    try:
        host_part()
    except Exception:
        import traceback
        rec("host_error", traceback.format_exc())
    rec("done", True)


if __name__ == "__main__":
    main()
