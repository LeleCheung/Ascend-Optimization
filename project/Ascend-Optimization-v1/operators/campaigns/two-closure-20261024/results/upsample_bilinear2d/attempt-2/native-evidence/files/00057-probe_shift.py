"""Probe: replace tl.gather with +-1 element shifted contiguous block loads.

For the 2x horizontal expansion the even/odd outputs need V[k-1] and V[k+1].
Instead of shuffling V in UB, load the *source* row blocks shifted by one
element and mix them (the mix is linear, so left(V) = 0.25*left(F)+0.75*left(G)).
Only the first/last element of a tile can fall outside the tensor, and those
columns are exactly the ones repaired by the row-start/row-end rule, so a
one-element mask is both safe and value-preserving.

Both the masked form (safe for a caller-owned tensor) and the unmasked form
(probe buffer is pre-padded) are measured so the cost of the mask is visible.
"""

import json
import os
import statistics
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = os.path.join(ART, "probe_shift.json")
res = {"rows": [], "errors": []}


def flush():
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)


@triton.jit
def k_gather(ptr_o, ptr_i,
             IH: tl.constexpr, IW: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
             NB: tl.constexpr, QB: tl.constexpr, H16: tl.constexpr):
    p = tl.program_id(0)
    img = p // NB
    b = p % NB
    r0 = b * QB
    lb = tl.maximum(r0 - 1, 0)
    L: tl.constexpr = QB * IW
    offs = tl.arange(0, L)
    ibase = img * (IH * IW) + lb * IW
    f = tl.load(ptr_i + ibase + offs)
    g = tl.load(ptr_i + ibase + IW + offs)
    if H16:
        vf = tl.reshape(0.25 * f + 0.75 * g, (QB, IW))
        vw = tl.reshape(0.75 * f + 0.25 * g, (QB, IW))
    else:
        vf = tl.reshape(0.25 * f.to(tl.float32) + 0.75 * g.to(tl.float32), (QB, IW))
        vw = tl.reshape(0.75 * f.to(tl.float32) + 0.25 * g.to(tl.float32), (QB, IW))
    k = tl.arange(0, IW)
    il = tl.broadcast_to(tl.maximum(k - 1, 0)[None, :], (QB, IW))
    ir = tl.broadcast_to(tl.minimum(k + 1, IW - 1)[None, :], (QB, IW))
    ef = 0.25 * tl.gather(vf, il, 1) + 0.75 * vf
    of = 0.75 * vf + 0.25 * tl.gather(vf, ir, 1)
    ew = 0.25 * tl.gather(vw, il, 1) + 0.75 * vw
    ow = 0.75 * vw + 0.25 * tl.gather(vw, ir, 1)
    je = tl.reshape(tl.join(ef, of), (QB, OW))
    jo = tl.reshape(tl.join(ew, ow), (QB, OW))
    oo = (2 * (lb + tl.arange(0, QB)) * OW)[:, None] + tl.arange(0, OW)[None, :]
    obase = img * (OH * OW)
    tl.store(ptr_o + obase + oo + 2 * OW, je.to(ptr_o.dtype.element_ty))
    tl.store(ptr_o + obase + oo + OW, jo.to(ptr_o.dtype.element_ty))


@triton.jit
def k_shift(ptr_o, ptr_i,
            IH: tl.constexpr, IW: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
            NB: tl.constexpr, QB: tl.constexpr, H16: tl.constexpr, MASKED: tl.constexpr):
    p = tl.program_id(0)
    img = p // NB
    b = p % NB
    r0 = b * QB
    lb = tl.maximum(r0 - 1, 0)
    L: tl.constexpr = QB * IW
    offs = tl.arange(0, L)
    ibase = img * (IH * IW) + lb * IW
    if MASKED:
        ml = offs >= 1
        mr = offs < L - 1
        f = tl.load(ptr_i + ibase + offs)
        g = tl.load(ptr_i + ibase + IW + offs)
        lf = tl.load(ptr_i + ibase - 1 + offs, mask=ml, other=0.0)
        lg = tl.load(ptr_i + ibase + IW - 1 + offs, mask=ml, other=0.0)
        rf = tl.load(ptr_i + ibase + 1 + offs, mask=mr, other=0.0)
        rg = tl.load(ptr_i + ibase + IW + 1 + offs, mask=mr, other=0.0)
    else:
        f = tl.load(ptr_i + ibase + offs)
        g = tl.load(ptr_i + ibase + IW + offs)
        lf = tl.load(ptr_i + ibase - 1 + offs)
        lg = tl.load(ptr_i + ibase + IW - 1 + offs)
        rf = tl.load(ptr_i + ibase + 1 + offs)
        rg = tl.load(ptr_i + ibase + IW + 1 + offs)

    if H16:
        vf = 0.25 * f + 0.75 * g
        vw = 0.75 * f + 0.25 * g
        ef = 0.25 * (0.25 * lf + 0.75 * lg) + 0.75 * vf
        ew = 0.25 * (0.75 * lf + 0.25 * lg) + 0.75 * vw
        of = 0.75 * vf + 0.25 * (0.25 * rf + 0.75 * rg)
        ow = 0.75 * vw + 0.25 * (0.75 * rf + 0.25 * rg)
    else:
        ff = f.to(tl.float32)
        gg = g.to(tl.float32)
        lff = lf.to(tl.float32)
        lgg = lg.to(tl.float32)
        rff = rf.to(tl.float32)
        rgg = rg.to(tl.float32)
        vf = 0.25 * ff + 0.75 * gg
        vw = 0.75 * ff + 0.25 * gg
        ef = 0.25 * (0.25 * lff + 0.75 * lgg) + 0.75 * vf
        ew = 0.25 * (0.75 * lff + 0.25 * lgg) + 0.75 * vw
        of = 0.75 * vf + 0.25 * (0.25 * rff + 0.75 * rgg)
        ow = 0.75 * vw + 0.25 * (0.75 * rff + 0.25 * rgg)

    c0 = (offs % IW) == 0
    cN = (offs % IW) == IW - 1
    ef = tl.where(c0, vf, ef)
    ew = tl.where(c0, vw, ew)
    of = tl.where(cN, vf, of)
    ow = tl.where(cN, vw, ow)

    je = tl.reshape(tl.join(tl.reshape(ef, (QB, IW)), tl.reshape(of, (QB, IW))), (QB, OW))
    jo = tl.reshape(tl.join(tl.reshape(ew, (QB, IW)), tl.reshape(ow, (QB, IW))), (QB, OW))
    oo = (2 * (lb + tl.arange(0, QB)) * OW)[:, None] + tl.arange(0, OW)[None, :]
    obase = img * (OH * OW)
    tl.store(ptr_o + obase + oo + 2 * OW, je.to(ptr_o.dtype.element_ty))
    tl.store(ptr_o + obase + oo + OW, jo.to(ptr_o.dtype.element_ty))


@triton.jit
def k_shift_loadonly(ptr_o, ptr_i,
                     IH: tl.constexpr, IW: tl.constexpr, OH: tl.constexpr,
                     NB: tl.constexpr, QB: tl.constexpr):
    p = tl.program_id(0)
    img = p // NB
    b = p % NB
    lb = tl.maximum(b * QB - 1, 0)
    L: tl.constexpr = QB * IW
    offs = tl.arange(0, L)
    ibase = img * (IH * IW) + lb * IW
    ml = offs >= 1
    mr = offs < L - 1
    f = tl.load(ptr_i + ibase + offs).to(tl.float32)
    g = tl.load(ptr_i + ibase + IW + offs).to(tl.float32)
    lf = tl.load(ptr_i + ibase - 1 + offs, mask=ml, other=0.0).to(tl.float32)
    lg = tl.load(ptr_i + ibase + IW - 1 + offs, mask=ml, other=0.0).to(tl.float32)
    rf = tl.load(ptr_i + ibase + 1 + offs, mask=mr, other=0.0).to(tl.float32)
    rg = tl.load(ptr_i + ibase + IW + 1 + offs, mask=mr, other=0.0).to(tl.float32)
    v = 0.25 * f + 0.75 * g + lf + lg + rf + rg
    tl.store(ptr_o + img * OH + lb + tl.arange(0, QB),
             tl.sum(tl.reshape(v, (QB, IW)), 1).to(ptr_o.dtype.element_ty))


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
    ("t5_fp32_large", torch.float32, (4, 32, 128, 128), (256, 256)),
]

for name, dtype, shape, out_size in CASES:
    N, C, IH, IW = shape
    OH, OW = out_size
    n = N * C * IH * IW
    buf = torch.empty(n + 8, dtype=dtype, device="npu")
    g0 = torch.Generator().manual_seed(1730)
    buf[4:4 + n] = torch.randn(n, generator=g0).to(dtype=dtype, device="npu")
    x = buf[4:4 + n].view(shape)
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
                # block b covers output rows [2*lb+1, 2*lb+2qb]; check everything
                # the block kernel is responsible for (rows 1..OH-1)
                rec["max_abs_err"] = float(
                    (out[:, :, 1:-1].float() - ref[:, :, 1:-1].float()).abs().max())
            row["variants"].append(rec)
        except Exception as exc:
            row["variants"].append({"variant": vname, "error": repr(exc)[:400]})
        flush()

    for qb in (16, 32):
        if IH % qb or IH // qb < 1:
            continue
        nb = IH // qb
        grid = (N * C * nb,)
        h16 = dtype is torch.float16
        add(f"gather_qb{qb}", lambda qb=qb, nb=nb, grid=grid, h16=h16: k_gather[grid](
            out, x, IH=IH, IW=IW, OH=OH, OW=OW, NB=nb, QB=qb, H16=h16), qb=qb)
        add(f"shift_mask_qb{qb}", lambda qb=qb, nb=nb, grid=grid, h16=h16: k_shift[grid](
            out, x, IH=IH, IW=IW, OH=OH, OW=OW, NB=nb, QB=qb, H16=h16, MASKED=True), qb=qb)
        add(f"shift_nomask_qb{qb}", lambda qb=qb, nb=nb, grid=grid, h16=h16: k_shift[grid](
            out, x, IH=IH, IW=IW, OH=OH, OW=OW, NB=nb, QB=qb, H16=h16, MASKED=False), qb=qb)
        add(f"shift_f32math_qb{qb}", lambda qb=qb, nb=nb, grid=grid: k_shift[grid](
            out, x, IH=IH, IW=IW, OH=OH, OW=OW, NB=nb, QB=qb, H16=False, MASKED=True), qb=qb)
        add(f"shift_loadonly_qb{qb}", lambda qb=qb, nb=nb, grid=grid: k_shift_loadonly[grid](
            out, x, IH=IH, IW=IW, OH=OH, NB=nb, QB=qb))
    res["rows"].append(row)
    flush()

flush()
print(json.dumps(res, indent=1)[:3000])
