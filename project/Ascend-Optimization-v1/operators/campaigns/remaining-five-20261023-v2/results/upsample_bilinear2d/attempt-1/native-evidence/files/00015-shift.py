"""Replace the misaligned (+-1 element) row loads with register-side shifts.

Evidence so far (t1 geometry, 8.4M outputs, BR=8, 4096 programs):
    k2  (2 aligned row loads + join + 3-D store)  0.594 ms
    k6  (the same plus 4 loads at +-1 offset)    21.31 ms
    store path only                               0.293 ms
So one misaligned contiguous load costs ~5 ms where an aligned one costs
~0.2 ms.  This probe tries to obtain v[k-1] / v[k+1] without any global
misaligned access, and also pins down how the penalty depends on the offset.
"""

import json
import os
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
def _rows(ptr_i, rows, s_n, s_c, s_h, rh,
          C: tl.constexpr, IH: tl.constexpr, OH: tl.constexpr):
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
def _st(ptr_o, val, rows, ka, BR: tl.constexpr, IWP: tl.constexpr, OW: tl.constexpr):
    ooff = (rows * OW)[:, None, None] + (ka * 2)[:, :, None] + tl.arange(0, 2)[None, None, :]
    tl.store(ptr_o + ooff, val.to(ptr_o.dtype.element_ty))


# A: two UB gathers replace the two misaligned load pairs
@triton.jit
def kga(ptr_o, ptr_i, n_rows, s_n, s_c, s_h, rh,
        C: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
        OH: tl.constexpr, OW: tl.constexpr, BR: tl.constexpr, IWP: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rr0, rr1, hl = _rows(ptr_i, rows, s_n, s_c, s_h, rh, C, IH, OH)
    ka = tl.arange(0, IWP)[None, :]
    h0l = 1.0 - hl
    v = h0l * tl.load(ptr_i + (rr0 + ka)).to(tl.float32) \
        + hl * tl.load(ptr_i + (rr1 + ka)).to(tl.float32)
    k = tl.arange(0, IWP)
    il = tl.broadcast_to(tl.maximum(k - 1, 0)[None, :], (BR, IWP))
    ir = tl.broadcast_to(tl.minimum(k + 1, IW - 1)[None, :], (BR, IWP))
    vm = tl.gather(v, il, 1)
    vp = tl.gather(v, ir, 1)
    _st(ptr_o, tl.join(0.25 * vm + 0.75 * v, 0.75 * v + 0.25 * vp),
        rows, ka, BR, IWP, OW)


# B: one gather only (identity index) -- isolates gather cost from the maths
@triton.jit
def kgb(ptr_o, ptr_i, n_rows, s_n, s_c, s_h, rh,
        C: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
        OH: tl.constexpr, OW: tl.constexpr, BR: tl.constexpr, IWP: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rr0, rr1, hl = _rows(ptr_i, rows, s_n, s_c, s_h, rh, C, IH, OH)
    ka = tl.arange(0, IWP)[None, :]
    h0l = 1.0 - hl
    v = h0l * tl.load(ptr_i + (rr0 + ka)).to(tl.float32) \
        + hl * tl.load(ptr_i + (rr1 + ka)).to(tl.float32)
    k = tl.arange(0, IWP)
    ir = tl.broadcast_to(tl.minimum(k + 1, IW - 1)[None, :], (BR, IWP))
    vp = tl.gather(v, ir, 1)
    _st(ptr_o, tl.join(0.75 * v, 0.75 * v + 0.25 * vp), rows, ka, BR, IWP, OW)


# C: misalignment granularity -- add one load shifted by OFF elements
@triton.jit
def koff(ptr_o, ptr_i, n_rows, s_n, s_c, s_h, rh, OFF: tl.constexpr,
         C: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
         OH: tl.constexpr, OW: tl.constexpr, BR: tl.constexpr, IWP: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rr0, rr1, hl = _rows(ptr_i, rows, s_n, s_c, s_h, rh, C, IH, OH)
    ka = tl.arange(0, IWP)[None, :]
    h0l = 1.0 - hl
    v = h0l * tl.load(ptr_i + (rr0 + ka)).to(tl.float32) \
        + hl * tl.load(ptr_i + (rr1 + ka)).to(tl.float32)
    d = h0l * tl.load(ptr_i + (rr0 + ka + OFF)).to(tl.float32) \
        + hl * tl.load(ptr_i + (rr1 + ka + OFF)).to(tl.float32)
    _st(ptr_o, tl.join(0.25 * d + 0.75 * v, 0.75 * v + 0.25 * d),
        rows, ka, BR, IWP, OW)


# D: strided (stride 2) load -- is a non-unit but aligned-stride pattern cheap?
@triton.jit
def kstr(ptr_o, ptr_i, n_rows, s_n, s_c, s_h, rh,
         C: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
         OH: tl.constexpr, OW: tl.constexpr, BR: tl.constexpr, IWP: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rr0, rr1, hl = _rows(ptr_i, rows, s_n, s_c, s_h, rh, C, IH, OH)
    ka = tl.arange(0, IWP)[None, :]
    h0l = 1.0 - hl
    v = h0l * tl.load(ptr_i + (rr0 + ka)).to(tl.float32) \
        + hl * tl.load(ptr_i + (rr1 + ka)).to(tl.float32)
    d = h0l * tl.load(ptr_i + (rr0 + ka * 2)).to(tl.float32) \
        + hl * tl.load(ptr_i + (rr1 + ka * 2)).to(tl.float32)
    _st(ptr_o, tl.join(0.25 * d + 0.75 * v, 0.75 * v + 0.25 * d),
        rows, ka, BR, IWP, OW)


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
    # over-allocate so shifted reads stay inside the allocation
    xpad = torch.empty(N * C * IH * IW + 64, device=DEV, dtype=torch.float16)
    xpad[32:-32] = x.reshape(-1)
    xp = xpad[32:-32]
    outp = torch.empty((N, C, OH, OW), device=DEV, dtype=torch.float16)
    n_rows = N * C * OH
    grid = (n_rows // BR,)
    rh = IH / OH
    base = dict(C=C, IH=IH, IW=IW, OH=OH, OW=OW, BR=BR, IWP=IWP)
    res = {}
    for name, k, ptr, extra in [
        ("A_2gather", kga, x, {}),
        ("B_1gather", kgb, x, {}),
        ("C_off1", koff, xp, {"OFF": 1}),
        ("C_off2", koff, xp, {"OFF": 2}),
        ("C_off8", koff, xp, {"OFF": 8}),
        ("C_off16", koff, xp, {"OFF": 16}),
        ("D_stride2", kstr, xp, {}),
    ]:
        try:
            res[name] = bench_call(lambda k=k, ptr=ptr, extra=extra: k[grid](
                outp, ptr, n_rows, *x.stride()[:3], rh, **extra, **base))
        except Exception:
            res[name] = "ERR:" + traceback.format_exc().splitlines()[-1][:200]
    out = {"ms": res, "ref_baseline_ms": {"k2_2aligned_loads": 0.5944}}
    with open(os.path.join(ART, "shift.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


main()
