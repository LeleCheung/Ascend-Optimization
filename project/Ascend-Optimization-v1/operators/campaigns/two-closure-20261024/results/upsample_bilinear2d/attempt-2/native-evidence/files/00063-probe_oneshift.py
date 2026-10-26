"""Probe a one-shift horizontal expansion (2 gathers instead of 4).

Because the odd output at column k and the even output at column k+1 are both
mixtures of the same pair (V[k], V[k+1]), the interleaved output row can be
written as interleave(A, B) placed one *element* higher, with
    A[k] = 0.75 V[k] + 0.25 V[k+1]   (odd column k  -> position 2k+1)
    B[k] = 0.25 V[k] + 0.75 V[k+1]   (even column k+1 -> position 2k+2)
so only the +1 neighbour is needed.  Consequences: the tile's last element
falls past the row, hence a tail mask, and element 0 of every row is written
by a separate per-row store of V's column 0 (which is exactly the reference
value there: w0 = 0 makes the horizontal pair 1.0/0.0).

Interior blocks only (row 0 and OH-1 excluded) so both variants are compared
over identical rows; numerics are checked on rows 1 .. OH-2.
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
OUT = os.path.join(ART, "probe_oneshift.json")
res = {"rows": [], "errors": []}


def flush():
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)


@triton.jit
def k4g(ptr_o, ptr_i, IH: tl.constexpr, IW: tl.constexpr, OH: tl.constexpr,
        OW: tl.constexpr, NB: tl.constexpr, QB: tl.constexpr, H16: tl.constexpr):
    pid = tl.program_id(0)
    img = pid // NB
    b = pid - img * NB
    r0 = b * QB
    lb = tl.maximum(r0 - 1, 0)
    L: tl.constexpr = QB * IW
    offs = tl.arange(0, L)
    ibase = img * (IH * IW) + lb * IW
    f = tl.load(ptr_i + ibase + offs)
    g = tl.load(ptr_i + ibase + IW + offs)
    if H16:
        vf = tl.reshape(0.25 * f + 0.75 * g, (QB, IW))
        vo = tl.reshape(0.75 * f + 0.25 * g, (QB, IW))
    else:
        vf = tl.reshape(0.25 * f.to(tl.float32) + 0.75 * g.to(tl.float32), (QB, IW))
        vo = tl.reshape(0.75 * f.to(tl.float32) + 0.25 * g.to(tl.float32), (QB, IW))
    k = tl.arange(0, IW)
    il = tl.broadcast_to(tl.maximum(k - 1, 0)[None, :], (QB, IW))
    ir = tl.broadcast_to(tl.minimum(k + 1, IW - 1)[None, :], (QB, IW))
    ef = 0.25 * tl.gather(vf, il, 1) + 0.75 * vf
    of = 0.75 * vf + 0.25 * tl.gather(vf, ir, 1)
    eo = 0.25 * tl.gather(vo, il, 1) + 0.75 * vo
    oo_ = 0.75 * vo + 0.25 * tl.gather(vo, ir, 1)
    pe = tl.reshape(tl.join(ef, of), (QB, OW))
    po = tl.reshape(tl.join(eo, oo_), (QB, OW))
    oo = (2 * (lb + tl.arange(0, QB)) * OW)[:, None] + tl.arange(0, OW)[None, :]
    obase = img * (OH * OW)
    tl.store(ptr_o + obase + oo + 2 * OW, pe.to(ptr_o.dtype.element_ty))
    tl.store(ptr_o + obase + oo + OW, po.to(ptr_o.dtype.element_ty))


@triton.jit
def k1s(ptr_o, ptr_i, IH: tl.constexpr, IW: tl.constexpr, OH: tl.constexpr,
        OW: tl.constexpr, NB: tl.constexpr, QB: tl.constexpr, H16: tl.constexpr,
        FIX: tl.constexpr, TAILMASK: tl.constexpr):
    pid = tl.program_id(0)
    img = pid // NB
    b = pid - img * NB
    r0 = b * QB
    lb = tl.maximum(r0 - 1, 0)
    L: tl.constexpr = QB * IW
    offs = tl.arange(0, L)
    ibase = img * (IH * IW) + lb * IW
    f = tl.load(ptr_i + ibase + offs)
    g = tl.load(ptr_i + ibase + IW + offs)
    if H16:
        vf = tl.reshape(0.25 * f + 0.75 * g, (QB, IW))
        vo = tl.reshape(0.75 * f + 0.25 * g, (QB, IW))
    else:
        vf = tl.reshape(0.25 * f.to(tl.float32) + 0.75 * g.to(tl.float32), (QB, IW))
        vo = tl.reshape(0.75 * f.to(tl.float32) + 0.25 * g.to(tl.float32), (QB, IW))
    k = tl.arange(0, IW)
    ir = tl.broadcast_to(tl.minimum(k + 1, IW - 1)[None, :], (QB, IW))
    vr = tl.gather(vf, ir, 1)
    vq = tl.gather(vo, ir, 1)
    af = 0.75 * vf + 0.25 * vr
    bf = 0.25 * vf + 0.75 * vr
    ao = 0.75 * vo + 0.25 * vq
    bo = 0.25 * vo + 0.75 * vq
    pe = tl.reshape(tl.join(af, bf), (QB, OW))
    po = tl.reshape(tl.join(ao, bo), (QB, OW))
    c = tl.arange(0, OW)
    obase = img * (OH * OW)
    rowb = (2 * (lb + tl.arange(0, QB)) * OW)[:, None]
    if TAILMASK:
        m = tl.broadcast_to((c < OW - 1)[None, :], (QB, OW))
        tl.store(ptr_o + obase + rowb + 2 * OW + 1 + c[None, :],
                 pe.to(ptr_o.dtype.element_ty), mask=m)
        tl.store(ptr_o + obase + rowb + OW + 1 + c[None, :],
                 po.to(ptr_o.dtype.element_ty), mask=m)
    else:
        tl.store(ptr_o + obase + rowb + 2 * OW + 1 + c[None, :],
                 pe.to(ptr_o.dtype.element_ty))
        tl.store(ptr_o + obase + rowb + OW + 1 + c[None, :],
                 po.to(ptr_o.dtype.element_ty))
    if FIX:
        c2 = tl.broadcast_to((k == 0)[None, :], (QB, IW))
        c0f = tl.sum(tl.where(c2, vf, 0.0), 1)
        c0o = tl.sum(tl.where(c2, vo, 0.0), 1)
        tl.store(ptr_o + obase + rowb[:, 0] + 2 * OW, c0f.to(ptr_o.dtype.element_ty))
        tl.store(ptr_o + obase + rowb[:, 0] + OW, c0o.to(ptr_o.dtype.element_ty))


def bench(fn, reps=20, iters=5):
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
    return statistics.median(samples)


CASES = [
    ("t1_fp16", torch.float16, (4, 32, 128, 128), (256, 256)),
    ("t5_fp32", torch.float32, (4, 32, 128, 128), (256, 256)),
    ("t0_fp16", torch.float16, (1, 16, 64, 64), (128, 128)),
]

for name, dtype, shape, size in CASES:
    N, C, IH, IW = shape
    OH, OW = size
    g = torch.Generator().manual_seed(1730)
    x = torch.randn(shape, generator=g).to(dtype=dtype, device="npu")
    ref = torch.ops.aten.upsample_bilinear2d(x, [OH, OW], False, None, None)
    out = torch.empty((N, C, OH, OW), device="npu", dtype=dtype)
    half = dtype is torch.float16
    row = {"case": name, "variants": []}
    for qb in (16, 32):
        if IH % qb or IH // qb < 1:
            continue
        nb = IH // qb
        grid = (N * C * nb,)

        def add(vname, fn):
            try:
                out.zero_()
                fn()
                torch.npu.synchronize()
                got = out[:, :, 1:-1].float()
                exp = ref[:, :, 1:-1].float()
                err = float((got - exp).abs().max())
                ms = bench(fn)
                row["variants"].append({"variant": vname, "median_ms": ms,
                                        "max_abs_err": err})
            except Exception as exc:
                row["variants"].append({"variant": vname, "error": repr(exc)[:300]})
            flush()

        add(f"k4g_qb{qb}", lambda qb=qb, nb=nb, grid=grid: k4g[grid](
            out, x, IH=IH, IW=IW, OH=OH, OW=OW, NB=nb, QB=qb, H16=half))
        add(f"k1s_qb{qb}", lambda qb=qb, nb=nb, grid=grid: k1s[grid](
            out, x, IH=IH, IW=IW, OH=OH, OW=OW, NB=nb, QB=qb, H16=half,
            FIX=True, TAILMASK=True))
        add(f"k1s_nofix_qb{qb}", lambda qb=qb, nb=nb, grid=grid: k1s[grid](
            out, x, IH=IH, IW=IW, OH=OH, OW=OW, NB=nb, QB=qb, H16=half,
            FIX=False, TAILMASK=True))
        add(f"k1s_nomask_qb{qb}", lambda qb=qb, nb=nb, grid=grid: k1s[grid](
            out, x, IH=IH, IW=IW, OH=OH, OW=OW, NB=nb, QB=qb, H16=half,
            FIX=True, TAILMASK=False))
    res["rows"].append(row)
    flush()

flush()
print(json.dumps(res, indent=1)[:4000])
