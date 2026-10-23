"""Localise the per-element cost of the current gather-based kernel.

Measures, on the real target, several kernel shapes for the two timing
geometries so we can tell apart three competing explanations:

  (a) per-program launch/teardown overhead dominates,
  (b) per-lane (non-affine) vector-of-pointers loads are scalarised,
  (c) the register-level gather / interleave primitives are themselves slow.

Nothing here is a candidate; it only produces diagnostic timings.
"""

import json
import os
import statistics
import traceback

import torch
import torch_npu  # noqa: F401  registers torch.npu

import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
DEV = "npu:0"


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    evs = [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
           for _ in range(iters)]
    for s, e in evs:
        s.record()
        fn()
        e.record()
    torch.npu.synchronize()
    vals = sorted(s.elapsed_time(e) for s, e in evs)
    return statistics.median(vals)


# ---------------------------------------------------------------- primitives
@triton.jit
def k_touch(ptr_o):
    tl.store(ptr_o + tl.arange(0, 8), tl.zeros([8], dtype=tl.float32))


@triton.jit
def k_copy(ptr_o, ptr_i, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr_o + offs, tl.load(ptr_i + offs))


@triton.jit
def k_gather1(ptr_o, ptr_i, BLOCK: tl.constexpr, IWP: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(ptr_i + (offs % IWP))
    tl.store(ptr_o + offs, v)


@triton.jit
def k_gather4(ptr_o, ptr_i, BLOCK: tl.constexpr, IWP: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    idx = offs % IWP
    v0 = tl.load(ptr_i + idx)
    v1 = tl.load(ptr_i + ((idx + 1) % IWP))
    v2 = tl.load(ptr_i + ((idx + 2) % IWP))
    v3 = tl.load(ptr_i + ((idx + 3) % IWP))
    v = v0.to(tl.float32) * 0.25 + v1.to(tl.float32) * 0.25 \
        + v2.to(tl.float32) * 0.25 + v3.to(tl.float32) * 0.25
    tl.store(ptr_o + offs, v.to(ptr_o.dtype.element_ty))


# row based: one output row per program, contiguous input row loads
@triton.jit
def k_row_gather(ptr_o, ptr_i, rh, rw,
                 IH: tl.constexpr, IW: tl.constexpr,
                 OH: tl.constexpr, OW: tl.constexpr,
                 IWP: tl.constexpr, OWP: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % OH
    nc = pid // OH
    real_h = tl.maximum((oh.to(tl.float32) + 0.5) * rh - 0.5, 0.0)
    h0f = real_h.to(tl.int32)
    hl = real_h - h0f.to(tl.float32)
    h0 = tl.minimum(h0f, IH - 1)
    h1 = tl.minimum(h0 + 1, IH - 1)
    ow = tl.arange(0, OWP)
    real_w = tl.maximum((ow.to(tl.float32) + 0.5) * rw - 0.5, 0.0)
    w0f = real_w.to(tl.int32)
    wl = real_w - w0f.to(tl.float32)
    w0 = tl.minimum(w0f, IW - 1)
    w1 = tl.minimum(w0 + 1, IW - 1)
    base = nc * (IH * IW)
    k = tl.arange(0, IWP)
    r0 = tl.load(ptr_i + base + h0 * IW + k)
    r1 = tl.load(ptr_i + base + h1 * IW + k)
    v00 = tl.gather(r0, w0, 0)
    v01 = tl.gather(r0, w1, 0)
    v10 = tl.gather(r1, w0, 0)
    v11 = tl.gather(r1, w1, 0)
    w0l = 1.0 - wl
    h0l = 1.0 - hl
    top = w0l * v00 + wl * v01
    bot = w0l * v10 + wl * v11
    val = h0l * top + hl * bot
    tl.store(ptr_o + pid * OW + ow, val.to(ptr_o.dtype.element_ty), mask=ow < OW)


# row based, general horizontal via gather but reading rows with a vector base
@triton.jit
def k_row2d(ptr_o, ptr_i, rh, rw,
            IH: tl.constexpr, IW: tl.constexpr,
            OH: tl.constexpr, OW: tl.constexpr,
            BR: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    oh = rows % OH
    nc = rows // OH
    real_h = tl.maximum((oh.to(tl.float32) + 0.5) * rh - 0.5, 0.0)
    h0 = tl.minimum(real_h.to(tl.int32), IH - 1)
    hl = real_h - h0.to(tl.float32)
    h1 = tl.minimum(h0 + 1, IH - 1)
    base = nc * (IH * IW)
    k = tl.arange(0, IW)
    off = (base + h0 * IW)[:, None] + k[None, :]
    v = tl.load(ptr_i + off)
    tl.store(ptr_o + (rows * IW)[:, None] + k[None, :], v)


def main():
    out = {}
    torch.npu.set_device(DEV)

    # ---- timing-0 geometry: (1,16,64,64) -> (128,128) fp16
    geo = {
        "small": dict(N=1, C=16, IH=64, IW=64, OH=128, OW=128, dt=torch.float16),
        "large": dict(N=4, C=32, IH=128, IW=128, OH=256, OW=256, dt=torch.float16),
    }
    res = {}
    for name, g in geo.items():
        N, C, IH, IW, OH, OW, dt = (g["N"], g["C"], g["IH"], g["IW"],
                                    g["OH"], g["OW"], g["dt"])
        inp = torch.randn((N, C, IH, IW), device=DEV, dtype=dt)
        outp = torch.empty((N, C, OH, OW), device=DEV, dtype=dt)
        total = N * C * OH * OW
        nrows = N * C * OH
        rh = IH / OH
        rw = IW / OW
        r = {}

        def rec(key, fn):
            try:
                r[key] = round(bench(fn), 4)
            except Exception:
                r[key] = "ERR:" + traceback.format_exc().splitlines()[-1][:200]

        rec("launch_only", lambda: k_touch[(1,)](outp))

        for BLK in (256, 1024, 4096):
            g_ = (triton.cdiv(total, BLK),)
            rec(f"copy_BLK{BLK}", lambda g_=g_, BLK=BLK:
                k_copy[g_](outp, inp, BLOCK=BLK))

        rec("copy_nw1", lambda: k_copy[(triton.cdiv(total, 1024),)](
            outp, inp, BLOCK=1024, num_warps=1))
        rec("copy_nw8", lambda: k_copy[(triton.cdiv(total, 1024),)](
            outp, inp, BLOCK=1024, num_warps=8))

        # gather probes with the same element count as the output
        for BLK in (1024, 4096):
            g_ = (triton.cdiv(total, BLK),)
            rec(f"gather1_BLK{BLK}", lambda g_=g_, BLK=BLK:
                k_gather1[g_](outp, inp, BLOCK=BLK, IWP=IW * IH))
            rec(f"gather4_BLK{BLK}", lambda g_=g_, BLK=BLK:
                k_gather4[g_](outp, inp, BLOCK=BLK, IWP=IW * IH))

        rec("rowgather_1row", lambda: k_row_gather[(nrows,)](
            outp, inp, rh, rw, IH=IH, IW=IW, OH=OH, OW=OW,
            IWP=triton.next_power_of_2(IW), OWP=triton.next_power_of_2(OW)))

        for BR in (4, 8):
            if nrows % BR == 0:
                rec(f"row2d_gatherbase_BR{BR}",
                    lambda BR=BR: k_row2d[(nrows // BR,)](
                        outp, inp, rh, rw, IH=IH, IW=IW, OH=OH, OW=OW, BR=BR))

        res[name] = r

    out["timings_ms"] = res
    out["note"] = ("all timings are torch.npu.Event medians over 20 iters; "
                   "device span for one kernel submission")
    path = os.path.join(ART, "probe.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


main()
