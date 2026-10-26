"""Probe replacing the tl.gather horizontal shuffle with CANN UB slice ops.

All shapes stay powers of two and every offset is static: the flat block T of
L = QB*IW elements is inserted at offset 1 of a 2L tile, so +-1 column
neighbours are static-offset slices.  Row-boundary wraps are repaired with a
where on the column index, which reproduces PyTorch's edge rule exactly
(wl = 0 at ow = 0, and i1p = IW-1 at the last column).

Variants are ordered safest-first; artifacts are flushed after each one.
"""

import json
import os
import statistics
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from triton.language.extra.cann import extension as ext

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = os.path.join(ART, "probe_slice.json")
res = {"rows": [], "errors": []}


def flush():
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)


@triton.jit
def k_gather16(ptr_o, ptr_i,
               IH: tl.constexpr, IW: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
               NB: tl.constexpr, QB: tl.constexpr):
    p = tl.program_id(0)
    img = p // (NB - 2)
    b = p % (NB - 2) + 1
    r0 = b * QB
    L: tl.constexpr = QB * IW
    offs = tl.arange(0, L)
    ibase = img * (IH * IW) + (r0 - 1) * IW
    f = tl.load(ptr_i + ibase + offs)
    g = tl.load(ptr_i + ibase + IW + offs)
    h = tl.load(ptr_i + ibase + 2 * IW + offs)
    vf = tl.reshape(0.25 * f + 0.75 * g, (QB, IW))
    vw = tl.reshape(0.75 * g + 0.25 * h, (QB, IW))
    k = tl.arange(0, IW)
    il = tl.broadcast_to(tl.maximum(k - 1, 0)[None, :], (QB, IW))
    ir = tl.broadcast_to(tl.minimum(k + 1, IW - 1)[None, :], (QB, IW))
    ef = 0.25 * tl.gather(vf, il, 1) + 0.75 * vf
    of = 0.75 * vf + 0.25 * tl.gather(vf, ir, 1)
    ew = 0.25 * tl.gather(vw, il, 1) + 0.75 * vw
    ow = 0.75 * vw + 0.25 * tl.gather(vw, ir, 1)
    pfe = tl.reshape(tl.reshape(tl.join(ef, of), (QB, OW)), (QB, OW))
    pfo = tl.reshape(tl.reshape(tl.join(ew, ow), (QB, OW)), (QB, OW))
    oo = (2 * (r0 + tl.arange(0, QB)) * OW)[:, None] + tl.arange(0, OW)[None, :]
    obase = img * (OH * OW)
    tl.store(ptr_o + obase + oo, pfe.to(ptr_o.dtype.element_ty))
    tl.store(ptr_o + obase + oo + OW, pfo.to(ptr_o.dtype.element_ty))


@triton.jit
def k_slice(ptr_o, ptr_i,
            IH: tl.constexpr, IW: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
            NB: tl.constexpr, QB: tl.constexpr, H16: tl.constexpr):
    p = tl.program_id(0)
    img = p // (NB - 2)
    b = p % (NB - 2) + 1
    r0 = b * QB
    L: tl.constexpr = QB * IW
    offs = tl.arange(0, L)
    ibase = img * (IH * IW) + (r0 - 1) * IW
    f = tl.load(ptr_i + ibase + offs)
    g = tl.load(ptr_i + ibase + IW + offs)
    h = tl.load(ptr_i + ibase + 2 * IW + offs)
    if H16:
        vf = 0.25 * f + 0.75 * g
        vw = 0.75 * g + 0.25 * h
    else:
        vf = 0.25 * f.to(tl.float32) + 0.75 * g.to(tl.float32)
        vw = 0.75 * g.to(tl.float32) + 0.25 * h.to(tl.float32)

    z2 = tl.zeros((2 * L,), vf.dtype)
    pf = ext.insert_slice(z2, vf, [1], [L], [1])
    pw = ext.insert_slice(z2, vw, [1], [L], [1])
    lf = ext.extract_slice(pf, [0], [L], [1])
    rf = ext.extract_slice(pf, [2], [L], [1])
    lw = ext.extract_slice(pw, [0], [L], [1])
    rw = ext.extract_slice(pw, [2], [L], [1])

    col = offs % IW
    first = col == 0
    last = col == IW - 1
    ef = tl.where(first, vf, 0.25 * lf + 0.75 * vf)
    of = tl.where(last, vf, 0.75 * vf + 0.25 * rf)
    ew = tl.where(first, vw, 0.25 * lw + 0.75 * vw)
    ow = tl.where(last, vw, 0.75 * vw + 0.25 * rw)

    je = tl.reshape(tl.join(tl.reshape(ef, (QB, IW)), tl.reshape(of, (QB, IW))), (QB, OW))
    jo = tl.reshape(tl.join(tl.reshape(ew, (QB, IW)), tl.reshape(ow, (QB, IW))), (QB, OW))
    oo = (2 * (r0 + tl.arange(0, QB)) * OW)[:, None] + tl.arange(0, OW)[None, :]
    obase = img * (OH * OW)
    tl.store(ptr_o + obase + oo, je.to(ptr_o.dtype.element_ty))
    tl.store(ptr_o + obase + oo + OW, jo.to(ptr_o.dtype.element_ty))


@triton.jit
def k_slice2d(ptr_o, ptr_i,
              IH: tl.constexpr, IW: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
              NB: tl.constexpr, QB: tl.constexpr, H16: tl.constexpr):
    """Same shift but with a 2-D (QB, 2*IW) padded tile and per-row slices."""
    p = tl.program_id(0)
    img = p // (NB - 2)
    b = p % (NB - 2) + 1
    r0 = b * QB
    L: tl.constexpr = QB * IW
    offs = tl.arange(0, L)
    ibase = img * (IH * IW) + (r0 - 1) * IW
    f = tl.load(ptr_i + ibase + offs)
    g = tl.load(ptr_i + ibase + IW + offs)
    h = tl.load(ptr_i + ibase + 2 * IW + offs)
    if H16:
        vf = tl.reshape(0.25 * f + 0.75 * g, (QB, IW))
        vw = tl.reshape(0.75 * g + 0.25 * h, (QB, IW))
    else:
        vf = tl.reshape(0.25 * f.to(tl.float32) + 0.75 * g.to(tl.float32), (QB, IW))
        vw = tl.reshape(0.75 * g.to(tl.float32) + 0.25 * h.to(tl.float32), (QB, IW))

    z2 = tl.zeros((QB, 2 * IW), vf.dtype)
    pf = ext.insert_slice(z2, vf, [0, 1], [QB, IW], [1, 1])
    pw = ext.insert_slice(z2, vw, [0, 1], [QB, IW], [1, 1])
    lf = ext.extract_slice(pf, [0, 0], [QB, IW], [1, 1])
    rf = ext.extract_slice(pf, [0, 2], [QB, IW], [1, 1])
    lw = ext.extract_slice(pw, [0, 0], [QB, IW], [1, 1])
    rw = ext.extract_slice(pw, [0, 2], [QB, IW], [1, 1])

    col = tl.arange(0, IW)[None, :]
    ef = tl.where(col == 0, vf, 0.25 * lf + 0.75 * vf)
    of = tl.where(col == IW - 1, vf, 0.75 * vf + 0.25 * rf)
    ew = tl.where(col == 0, vw, 0.25 * lw + 0.75 * vw)
    ow = tl.where(col == IW - 1, vw, 0.75 * vw + 0.25 * rw)

    je = tl.reshape(tl.join(ef, of), (QB, OW))
    jo = tl.reshape(tl.join(ew, ow), (QB, OW))
    oo = (2 * (r0 + tl.arange(0, QB)) * OW)[:, None] + tl.arange(0, OW)[None, :]
    obase = img * (OH * OW)
    tl.store(ptr_o + obase + oo, je.to(ptr_o.dtype.element_ty))
    tl.store(ptr_o + obase + oo + OW, jo.to(ptr_o.dtype.element_ty))


def bench(fn, reps=30, iters=5):
    fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - t0) / reps * 1e3)
    return statistics.median(samples), min(samples)


CASES = [
    ("t1_fp16_large", torch.float16, (4, 32, 128, 128), (256, 256)),
    ("t0_fp16_small", torch.float16, (1, 16, 64, 64), (128, 128)),
]

for name, dtype, shape, out_size in CASES:
    N, C, IH, IW = shape
    OH, OW = out_size
    g = torch.Generator().manual_seed(1730)
    x = torch.randn(shape, generator=g).to(dtype=dtype, device="npu")
    ref = torch.ops.aten.upsample_bilinear2d(x, [OH, OW], False, None, None)
    out = torch.empty((N, C, OH, OW), device="npu", dtype=dtype)
    row = {"case": name, "variants": []}

    def add(vname, fn, qb=None):
        try:
            out.zero_()
            fn()
            torch.npu.synchronize()
            med, mn = bench(fn)
            rec = {"variant": vname, "median_ms": med, "min_ms": mn}
            if qb is not None:
                lo, hi = 2 * qb, OH - 2 * qb
                rec["interior_max_abs_err"] = float(
                    (out[:, :, lo:hi].float() - ref[:, :, lo:hi].float()).abs().max())
            row["variants"].append(rec)
        except Exception as exc:
            row["variants"].append({"variant": vname, "error": repr(exc)[:500]})
        flush()

    qb = 16 if IH >= 32 else 8
    nb = IH // qb
    grid = (N * C * (nb - 2),)
    add("gather_h16", lambda: k_gather16[grid](
        out, x, IH=IH, IW=IW, OH=OH, OW=OW, NB=nb, QB=qb), qb=qb)
    add("slice_flat_f32", lambda: k_slice[grid](
        out, x, IH=IH, IW=IW, OH=OH, OW=OW, NB=nb, QB=qb, H16=False), qb=qb)
    add("slice_flat_h16", lambda: k_slice[grid](
        out, x, IH=IH, IW=IW, OH=OH, OW=OW, NB=nb, QB=qb, H16=True), qb=qb)
    add("slice_2d_f32", lambda: k_slice2d[grid](
        out, x, IH=IH, IW=IW, OH=OH, OW=OW, NB=nb, QB=qb, H16=False), qb=qb)
    add("slice_2d_h16", lambda: k_slice2d[grid](
        out, x, IH=IH, IW=IW, OH=OH, OW=OW, NB=nb, QB=qb, H16=True), qb=qb)
    res["rows"].append(row)
    flush()

flush()
print(json.dumps(res, indent=1)[:3000])
