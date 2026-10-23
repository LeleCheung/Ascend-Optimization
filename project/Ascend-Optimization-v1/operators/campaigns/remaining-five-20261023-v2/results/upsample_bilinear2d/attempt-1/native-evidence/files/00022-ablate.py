"""Ablate the fast-path kernel to find the construct that scalarises.

Geometry of timing-1: (4,32,128,128) -> (256,256) fp16, 8.4M outputs.
Each variant removes exactly one construct so the per-element cost can be
attributed.  The current kernel measures ~21.3 ms here (~2.5 ns/output).
"""

import json
import os
import statistics
import traceback

import torch
import torch_npu  # noqa: F401

import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
DEV = "npu:0"

N, C, IH, IW, OH, OW = 4, 32, 128, 128, 256, 256
BR = 8
IWP = 128


@triton.jit
def _hdr(ptr_i, rows, n_rows, s_n, s_c, s_h, rh,
         C: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr, OH: tl.constexpr):
    oh = rows % OH
    nc = rows // OH
    n = nc // C
    c = nc - n * C
    real_h = tl.maximum((oh.to(tl.float32) + 0.5) * rh - 0.5, 0.0)
    h0f = real_h.to(tl.int32)
    hl = real_h - h0f.to(tl.float32)
    h0 = tl.minimum(h0f, IH - 1)
    h1 = tl.minimum(h0 + 1, IH - 1)
    base = n * s_n + c * s_c
    return (base + h0 * s_h)[:, None], (base + h1 * s_h)[:, None], hl[:, None]


@triton.jit
def _finish(ptr_o, val, rows, ka, n_rows,
            BR: tl.constexpr, IWP: tl.constexpr, IW: tl.constexpr, OW: tl.constexpr,
            MASKED: tl.constexpr):
    ooff = (rows * OW)[:, None, None] + (ka * 2)[:, :, None] + tl.arange(0, 2)[None, None, :]
    if MASKED:
        om = tl.broadcast_to((ka < IW)[:, :, None] & (rows[:, None, None] < n_rows),
                             (BR, IWP, 2))
        tl.store(ptr_o + ooff, val.to(ptr_o.dtype.element_ty), mask=om)
    else:
        tl.store(ptr_o + ooff, val.to(ptr_o.dtype.element_ty))


# 1: 2 unmasked centre loads, no 3-tap, 3-D store
@triton.jit
def k2(ptr_o, ptr_i, n_rows, s_n, s_c, s_h, rh,
       C: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
       OH: tl.constexpr, OW: tl.constexpr, BR: tl.constexpr, IWP: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rr0, rr1, hl = _hdr(ptr_i, rows, n_rows, s_n, s_c, s_h, rh, C, IH, IW, OH)
    ka = tl.arange(0, IWP)[None, :]
    v = (1.0 - hl) * tl.load(ptr_i + (rr0 + ka)).to(tl.float32) \
        + hl * tl.load(ptr_i + (rr1 + ka)).to(tl.float32)
    _finish(ptr_o, tl.join(0.75 * v, 0.25 * v), rows, ka, n_rows, BR, IWP, IW, OW, False)


# 2: same but with the full 2-D mask
@triton.jit
def k2m(ptr_o, ptr_i, n_rows, s_n, s_c, s_h, rh,
        C: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
        OH: tl.constexpr, OW: tl.constexpr, BR: tl.constexpr, IWP: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rr0, rr1, hl = _hdr(ptr_i, rows, n_rows, s_n, s_c, s_h, rh, C, IH, IW, OH)
    ka = tl.arange(0, IWP)[None, :]
    lm = tl.broadcast_to(ka < IW, (BR, IWP))
    v = (1.0 - hl) * tl.load(ptr_i + (rr0 + ka), mask=lm, other=0.0).to(tl.float32) \
        + hl * tl.load(ptr_i + (rr1 + ka), mask=lm, other=0.0).to(tl.float32)
    _finish(ptr_o, tl.join(0.75 * v, 0.25 * v), rows, ka, n_rows, BR, IWP, IW, OW, False)


# 3: six unmasked loads (tensor over-allocated so +-1 stays in bounds)
@triton.jit
def k6(ptr_o, ptr_i, n_rows, s_n, s_c, s_h, rh,
       C: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
       OH: tl.constexpr, OW: tl.constexpr, BR: tl.constexpr, IWP: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rr0, rr1, hl = _hdr(ptr_i, rows, n_rows, s_n, s_c, s_h, rh, C, IH, IW, OH)
    ka = tl.arange(0, IWP)[None, :]
    h0l = 1.0 - hl
    v = h0l * tl.load(ptr_i + (rr0 + ka)).to(tl.float32) \
        + hl * tl.load(ptr_i + (rr1 + ka)).to(tl.float32)
    vm = h0l * tl.load(ptr_i + (rr0 + ka - 1)).to(tl.float32) \
        + hl * tl.load(ptr_i + (rr1 + ka - 1)).to(tl.float32)
    vp = h0l * tl.load(ptr_i + (rr0 + ka + 1)).to(tl.float32) \
        + hl * tl.load(ptr_i + (rr1 + ka + 1)).to(tl.float32)
    even = 0.25 * vm + 0.75 * v
    odd = 0.75 * v + 0.25 * vp
    _finish(ptr_o, tl.join(even, odd), rows, ka, n_rows, BR, IWP, IW, OW, False)


# 4: six masked loads == current candidate structure
@triton.jit
def k6m(ptr_o, ptr_i, n_rows, s_n, s_c, s_h, rh,
        C: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
        OH: tl.constexpr, OW: tl.constexpr, BR: tl.constexpr, IWP: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rr0, rr1, hl = _hdr(ptr_i, rows, n_rows, s_n, s_c, s_h, rh, C, IH, IW, OH)
    ka = tl.arange(0, IWP)[None, :]
    lm = tl.broadcast_to(ka < IW, (BR, IWP))
    km1 = tl.broadcast_to(ka >= 1, (BR, IWP))
    kp1 = tl.broadcast_to(ka <= IW - 2, (BR, IWP))
    h0l = 1.0 - hl
    v = h0l * tl.load(ptr_i + (rr0 + ka), mask=lm, other=0.0).to(tl.float32) \
        + hl * tl.load(ptr_i + (rr1 + ka), mask=lm, other=0.0).to(tl.float32)
    vm = h0l * tl.load(ptr_i + (rr0 + ka - 1), mask=lm & km1, other=0.0).to(tl.float32) \
        + hl * tl.load(ptr_i + (rr1 + ka - 1), mask=lm & km1, other=0.0).to(tl.float32)
    vp = h0l * tl.load(ptr_i + (rr0 + ka + 1), mask=lm & kp1, other=0.0).to(tl.float32) \
        + hl * tl.load(ptr_i + (rr1 + ka + 1), mask=lm & kp1, other=0.0).to(tl.float32)
    even = 0.25 * vm + 0.75 * v
    odd = 0.75 * v + 0.25 * vp
    _finish(ptr_o, tl.join(even, odd), rows, ka, n_rows, BR, IWP, IW, OW, False)


# 5: store path only (constant value, no loads)
@triton.jit
def kst(ptr_o, ptr_i, n_rows, s_n, s_c, s_h, rh,
        C: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
        OH: tl.constexpr, OW: tl.constexpr, BR: tl.constexpr, IWP: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    ka = tl.arange(0, IWP)[None, :]
    v = tl.full((BR, IWP), 1.0, tl.float32) * (rows.to(tl.float32))[:, None]
    _finish(ptr_o, tl.join(0.75 * v, 0.25 * v), rows, ka, n_rows, BR, IWP, IW, OW, False)


# 6: 2-D contiguous store (no join, no 3-D offsets)
@triton.jit
def k2d(ptr_o, ptr_i, n_rows, s_n, s_c, s_h, rh,
        C: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
        OH: tl.constexpr, OW: tl.constexpr, BR: tl.constexpr, IWP: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rr0, rr1, hl = _hdr(ptr_i, rows, n_rows, s_n, s_c, s_h, rh, C, IH, IW, OH)
    ka = tl.arange(0, IWP)[None, :]
    v = (1.0 - hl) * tl.load(ptr_i + (rr0 + ka)).to(tl.float32) \
        + hl * tl.load(ptr_i + (rr1 + ka)).to(tl.float32)
    # same value, but written as BR full output rows with a 2-D offset tensor
    w = tl.arange(0, OW)[None, :]
    half = tl.join(0.75 * v, 0.25 * v).reshape(BR, OW)
    tl.store(ptr_o + (rows * OW)[:, None] + w, half.to(ptr_o.dtype.element_ty))


# 7: load path only (2 unmasked loads, collapse to one value per row)
@triton.jit
def kld(ptr_o, ptr_i, n_rows, s_n, s_c, s_h, rh,
        C: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
        OH: tl.constexpr, OW: tl.constexpr, BR: tl.constexpr, IWP: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rr0, rr1, hl = _hdr(ptr_i, rows, n_rows, s_n, s_c, s_h, rh, C, IH, IW, OH)
    ka = tl.arange(0, IWP)[None, :]
    v = (1.0 - hl) * tl.load(ptr_i + (rr0 + ka)).to(tl.float32) \
        + hl * tl.load(ptr_i + (rr1 + ka)).to(tl.float32)
    tl.store(ptr_o + rows, tl.sum(v, 1).to(ptr_o.dtype.element_ty))


def bench_call(fn, iters=20):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    e0 = torch.npu.Event(enable_timing=True)
    e1 = torch.npu.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.npu.synchronize()
    return round(e0.elapsed_time(e1) / iters, 4)


def main():
    torch.npu.set_device(DEV)
    x = torch.randn((N, C, IH, IW), device=DEV, dtype=torch.float16)
    xpad = torch.empty(N * C * IH * IW + 4, device=DEV, dtype=torch.float16)
    xpad[2:-2] = x.reshape(-1)
    xp = xpad[2:-2]
    outp = torch.empty((N, C, OH, OW), device=DEV, dtype=torch.float16)
    small = torch.empty(N * C * OH, device=DEV, dtype=torch.float16)
    n_rows = N * C * OH
    grid = (n_rows // BR,)
    rh = IH / OH
    args = dict(C=C, IH=IH, IW=IW, OH=OH, OW=OW, BR=BR, IWP=IWP)
    res = {}
    for name, k, ptr, dst in [
        ("k2_center_nomask", k2, x, outp),
        ("k2_center_masked", k2m, x, outp),
        ("k6_3tap_nomask", k6, xp, outp),
        ("k6_3tap_masked", k6m, x, outp),
        ("k_store_only", kst, x, outp),
        ("k_2d_store_reshape", k2d, x, outp),
        ("k_load_only", kld, x, small),
    ]:
        try:
            res[name] = bench_call(lambda k=k, ptr=ptr, dst=dst: k[grid](
                dst, ptr, n_rows, *x.stride()[:3], rh, **args))
        except Exception:
            res[name] = "ERR:" + traceback.format_exc().splitlines()[-1][:200]
    out = {"ms": res,
           "note": "t1 geometry (4,32,128,128)->(256,256) fp16, BR=8, "
                   "4096 programs, 8.4M outputs; 50 iters averaged"}
    with open(os.path.join(ART, "ablate.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


main()
