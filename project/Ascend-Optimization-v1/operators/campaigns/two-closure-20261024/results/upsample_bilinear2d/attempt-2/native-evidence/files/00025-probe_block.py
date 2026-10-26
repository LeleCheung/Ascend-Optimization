"""Probe a contiguous-input-block architecture for the 2x bilinear fast path.

The current kernel issues two 256-byte row loads per output row with a
data-dependent base.  Here a program owns QB consecutive input rows of one
channel image and loads three overlapping *contiguous* blocks (rows r0-1..,
r0.., r0+1..), each QB*IW elements, i.e. one multi-KB transfer per block.

Interior blocks only (b in 1..NB-2) so no masking or clamping is needed; the
numerics check therefore only compares the rows this probe writes.  The goal
is the achievable device time, plus the split between loads, stores, the
horizontal shuffle and the arithmetic.
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
OUT = os.path.join(ART, "probe_block.json")
res = {"rows": [], "errors": []}


def flush():
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)


# --------------------------------------------------------------- kernels
@triton.jit
def k_block(ptr_o, ptr_i,
            IH: tl.constexpr, IW: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
            NB: tl.constexpr, QB: tl.constexpr, H16: tl.constexpr, GATHER: tl.constexpr):
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

    k = tl.arange(0, IW)
    if GATHER:
        il = tl.broadcast_to(tl.maximum(k - 1, 0)[None, :], (QB, IW))
        ir = tl.broadcast_to(tl.minimum(k + 1, IW - 1)[None, :], (QB, IW))
        ef = 0.25 * tl.gather(vf, il, 1) + 0.75 * vf
        of = 0.75 * vf + 0.25 * tl.gather(vf, ir, 1)
        ew = 0.25 * tl.gather(vw, il, 1) + 0.75 * vw
        ow = 0.75 * vw + 0.25 * tl.gather(vw, ir, 1)
    else:
        ef = 0.25 * vf + 0.75 * vf
        of = 0.75 * vf + 0.25 * vf
        ew = 0.25 * vw + 0.75 * vw
        ow = 0.75 * vw + 0.25 * vw

    pfe = tl.reshape(tl.join(ef, of), (QB, OW))
    pfo = tl.reshape(tl.join(ew, ow), (QB, OW))
    oo = (2 * (r0 + tl.arange(0, QB)) * OW)[:, None] + tl.arange(0, OW)[None, :]
    obase = img * (OH * OW)
    tl.store(ptr_o + obase + oo, pfe.to(ptr_o.dtype.element_ty))
    tl.store(ptr_o + obase + oo + OW, pfo.to(ptr_o.dtype.element_ty))


@triton.jit
def k_block_loadonly(ptr_o, ptr_i,
                     IH: tl.constexpr, IW: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
                     NB: tl.constexpr, QB: tl.constexpr):
    p = tl.program_id(0)
    img = p // (NB - 2)
    b = p % (NB - 2) + 1
    r0 = b * QB
    L: tl.constexpr = QB * IW
    offs = tl.arange(0, L)
    ibase = img * (IH * IW) + (r0 - 1) * IW
    f = tl.load(ptr_i + ibase + offs).to(tl.float32)
    g = tl.load(ptr_i + ibase + IW + offs).to(tl.float32)
    h = tl.load(ptr_i + ibase + 2 * IW + offs).to(tl.float32)
    v = 0.25 * f + 0.75 * g + 0.75 * g + 0.25 * h
    vv = tl.reshape(v, (QB, IW))
    tl.store(ptr_o + img * OH + r0 + tl.arange(0, QB), tl.sum(vv, 1).to(ptr_o.dtype.element_ty))


@triton.jit
def k_block_storeonly(ptr_o,
                      IH: tl.constexpr, IW: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
                      NB: tl.constexpr, QB: tl.constexpr):
    p = tl.program_id(0)
    img = p // (NB - 2)
    b = p % (NB - 2) + 1
    r0 = b * QB
    val = tl.full((QB, OW), 1.0, tl.float32)
    oo = (2 * (r0 + tl.arange(0, QB)) * OW)[:, None] + tl.arange(0, OW)[None, :]
    obase = img * (OH * OW)
    tl.store(ptr_o + obase + oo, val.to(ptr_o.dtype.element_ty))
    tl.store(ptr_o + obase + oo + OW, val.to(ptr_o.dtype.element_ty))


@triton.jit
def k_block_flatstore(ptr_o, ptr_i,
                      IH: tl.constexpr, IW: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
                      NB: tl.constexpr, QB: tl.constexpr, H16: tl.constexpr, GATHER: tl.constexpr):
    """Same as k_block but the two output rows are joined through the 2-element
    axis and stored as one contiguous 3-D affine block (QB, 2, OW)."""
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
    k = tl.arange(0, IW)
    if GATHER:
        il = tl.broadcast_to(tl.maximum(k - 1, 0)[None, :], (QB, IW))
        ir = tl.broadcast_to(tl.minimum(k + 1, IW - 1)[None, :], (QB, IW))
        ef = 0.25 * tl.gather(vf, il, 1) + 0.75 * vf
        of = 0.75 * vf + 0.25 * tl.gather(vf, ir, 1)
        ew = 0.25 * tl.gather(vw, il, 1) + 0.75 * vw
        ow = 0.75 * vw + 0.25 * tl.gather(vw, ir, 1)
    else:
        ef = 0.25 * vf + 0.75 * vf
        of = 0.75 * vf + 0.25 * vf
        ew = 0.25 * vw + 0.75 * vw
        ow = 0.75 * vw + 0.25 * vw
    # (QB, IW, 2) for each row set, then interleave rows via a (QB, 2, ...) index
    je = tl.reshape(tl.join(ef, of), (QB, OW))
    jo = tl.reshape(tl.join(ew, ow), (QB, OW))
    both = tl.join(je, jo)                       # (QB, OW, 2) -> element interleave, not row
    # store the element-interleaved tile at row-interleaved addresses is wrong,
    # so this variant only measures the cost of the join/permute shape.
    oo = (2 * (r0 + tl.arange(0, QB)) * OW)[:, None, None] \
        + tl.arange(0, OW)[None, :, None] + tl.arange(0, 2)[None, None, :] * OW
    tl.store(ptr_o + img * (OH * OW) + oo, both.to(ptr_o.dtype.element_ty))


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
    ("t0_fp16_small", torch.float16, (1, 16, 64, 64), (128, 128)),
    ("t1_fp16_large", torch.float16, (4, 32, 128, 128), (256, 256)),
    ("t5_fp32_large", torch.float32, (4, 32, 128, 128), (256, 256)),
]

for name, dtype, shape, out_size in CASES:
    try:
        N, C, IH, IW = shape
        OH, OW = out_size
        g = torch.Generator().manual_seed(1730)
        x = torch.randn(shape, generator=g).to(dtype=dtype, device="npu")
        ref = torch.ops.aten.upsample_bilinear2d(x, [OH, OW], False, None, None)
        out = torch.empty((N, C, OH, OW), device="npu", dtype=dtype)
        row = {"case": name, "shape": list(shape), "variants": []}

        def add(vname, fn, qb=None):
            try:
                out.zero_()
                fn()
                torch.npu.synchronize()
                med, mn = bench(fn)
                rec = {"variant": vname, "median_ms": med, "min_ms": mn}
                if qb is not None:
                    lo, hi = 2 * qb, OH - 2 * qb
                    got = out[:, :, lo:hi].float()
                    exp = ref[:, :, lo:hi].float()
                    rec["interior_max_abs_err"] = float((got - exp).abs().max())
                row["variants"].append(rec)
            except Exception as exc:
                row["variants"].append({"variant": vname, "error": repr(exc)[:400]})
            flush()

        for qb in (4, 8, 16):
            if IH % qb:
                continue
            nb = IH // qb
            if nb <= 2:
                continue
            grid = (N * C * (nb - 2),)
            for h16 in (False, True):
                if h16 and dtype is not torch.float16:
                    continue
                add(f"block_qb{qb}_h16{int(h16)}", lambda qb=qb, nb=nb, grid=grid, h16=h16: k_block[grid](
                    out, x, IH=IH, IW=IW, OH=OH, OW=OW, NB=nb, QB=qb, H16=h16, GATHER=True), qb=qb)
            add(f"block_qb{qb}_nogather", lambda qb=qb, nb=nb, grid=grid: k_block[grid](
                out, x, IH=IH, IW=IW, OH=OH, OW=OW, NB=nb, QB=qb, H16=(dtype is torch.float16),
                GATHER=False), qb=qb)
            add(f"block_qb{qb}_loadonly", lambda qb=qb, nb=nb, grid=grid: k_block_loadonly[grid](
                out, x, IH=IH, IW=IW, OH=OH, OW=OW, NB=nb, QB=qb))
            add(f"block_qb{qb}_storeonly", lambda qb=qb, nb=nb, grid=grid: k_block_storeonly[grid](
                out, IH=IH, IW=IW, OH=OH, OW=OW, NB=nb, QB=qb))
            add(f"block_qb{qb}_flat3d", lambda qb=qb, nb=nb, grid=grid: k_block_flatstore[grid](
                out, x, IH=IH, IW=IW, OH=OH, OW=OW, NB=nb, QB=qb,
                H16=(dtype is torch.float16), GATHER=True))
        res["rows"].append(row)
        flush()
    except Exception as exc:
        res["errors"].append(f"{name}: {repr(exc)[:600]}")
        flush()

flush()
print(json.dumps(res, indent=1)[:3000])
