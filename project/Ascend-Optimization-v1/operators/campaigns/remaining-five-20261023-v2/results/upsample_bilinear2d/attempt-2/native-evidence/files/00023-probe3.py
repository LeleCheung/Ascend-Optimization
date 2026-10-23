"""Probe 3: isolate the horizontal +-1 neighbour mechanism.

A: 2 aligned row loads + blend + 1 store (no neighbour at all)
B: A + 2 tl.gather (axis=1)
C: A + 4 extra row loads at column offsets -1/+1
D: flat copy with a 1-element offset (misalignment cost)
Same geometry (2048 programs, BR=16, IWP=128) as the measured 0.63 ms baseline.
"""

import json
import os
import time
import traceback

import torch
import triton
import triton.language as tl

RES = {}


def rec(k, v):
    RES[k] = v
    print("%s = %s" % (k, v), flush=True)


def adir():
    d = os.environ.get("KGS_DEBUG_ARTIFACTS", "")
    return d if d and os.path.isdir(d) else os.getcwd()


@triton.jit
def _core(ptr_o, ptr_i, s_n, s_c, s_h, rh, mode: tl.constexpr,
          C: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
          OH: tl.constexpr, OW: tl.constexpr, BR: tl.constexpr,
          IWP: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
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
    rr0 = (base + h0 * s_h)[:, None]
    rr1 = (base + h1 * s_h)[:, None]
    ka = tl.arange(0, IWP)[None, :]
    hl2 = hl[:, None]
    h0l2 = 1.0 - hl2

    v = h0l2 * tl.load(ptr_i + (rr0 + ka)).to(tl.float32) \
        + hl2 * tl.load(ptr_i + (rr1 + ka)).to(tl.float32)

    if mode == 0:
        val = tl.reshape(tl.join(v, v), (BR, 2 * IWP))
    elif mode == 1:
        il = tl.broadcast_to(tl.maximum(tl.arange(0, IWP) - 1, 0)[None, :], (BR, IWP))
        ir = tl.broadcast_to(tl.minimum(tl.arange(0, IWP) + 1, IW - 1)[None, :], (BR, IWP))
        even = 0.25 * tl.gather(v, il, 1) + 0.75 * v
        odd = 0.75 * v + 0.25 * tl.gather(v, ir, 1)
        val = tl.reshape(tl.join(even, odd), (BR, 2 * IWP))
    else:
        rl0 = tl.maximum(rr0 - 1, 0)
        rl1 = tl.maximum(rr1 - 1, 0)
        vl = h0l2 * tl.load(ptr_i + (rl0 + ka)).to(tl.float32) \
            + hl2 * tl.load(ptr_i + (rl1 + ka)).to(tl.float32)
        vr = h0l2 * tl.load(ptr_i + (rr0 + ka + 1)).to(tl.float32) \
            + hl2 * tl.load(ptr_i + (rr1 + ka + 1)).to(tl.float32)
        even = 0.25 * vl + 0.75 * v
        odd = 0.75 * v + 0.25 * vr
        val = tl.reshape(tl.join(even, odd), (BR, 2 * IWP))

    ooff = (rows * OW)[:, None] + tl.arange(0, 2 * IWP)[None, :]
    tl.store(ptr_o + ooff, val.to(ptr_o.dtype.element_ty))


@triton.jit
def _flat_off(ptr_o, ptr_i, OFF: tl.constexpr, N: tl.constexpr):
    pid = tl.program_id(0)
    m = pid * N + tl.arange(0, N)
    tl.store(ptr_o + m, tl.load(ptr_i + (m + OFF)))


def time_wall(fn, iters=10, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


DEV = "npu:0"
BIG = (4, 32, 128, 128)
BH, BW = 256, 256


def main():
    pad = torch.randn(4 * 32 * 128 * 128 + 64, device=DEV, dtype=torch.float16)
    x = pad[:4 * 32 * 128 * 128].view(BIG)
    N, C, IH, IW = BIG
    n_rows = N * C * BH
    out = torch.empty((N, C, BH, BW), device=DEV, dtype=torch.float16)
    s_n, s_c, s_h, _ = x.stride()
    for br in (16, 32):
        IWP = 128
        grid = (n_rows // br,)
        for mode, name in ((0, "A_noload"), (1, "B_gather"), (2, "C_shift")):
            def call(br=br, grid=grid, mode=mode):
                _core[grid](out, x, s_n, s_c, s_h, IH / BH, mode=mode, C=C, IH=IH,
                            IW=IW, OH=BH, OW=BW, BR=br, IWP=IWP)
            try:
                call()
                torch.npu.synchronize()
                rec("wall_%s_br%d" % (name, br), time_wall(call, iters=10))
            except Exception as exc:
                rec("err_%s_br%d" % (name, br), repr(exc)[:400])
    # check correctness of mode 2 against the reference on a small case
    try:
        xs = pad[:1 * 16 * 64 * 64].view(1, 16, 64, 64)
        o2 = torch.empty((1, 16, 128, 128), device=DEV, dtype=torch.float16)
        sn, sc, sh, _ = xs.stride()
        _core[(1 * 16 * 128 // 16,)](o2, xs, sn, sc, sh, 64 / 128, mode=2, C=16,
                                     IH=64, IW=64, OH=128, OW=128, BR=16, IWP=64)
        torch.npu.synchronize()
        ref = torch.ops.aten.upsample_bilinear2d(xs, [128, 128], False, None, None)
        rec("maxdiff_C_small", (o2.float() - ref.float()).abs().max().item())
    except Exception as exc:
        rec("err_check_C", repr(exc)[:400])

    n = 1 << 23
    src = torch.randn(n + 64, device=DEV, dtype=torch.float16)
    dst = torch.empty(n, device=DEV, dtype=torch.float16)
    for off in (0, 1):
        for R in (32,):
            Nn = R * 128
            g = (n // Nn,)
            try:
                rec("wall_flat_off%d_R%d" % (off, R),
                    time_wall(lambda g=g, Nn=Nn, off=off: _flat_off[g](dst, src, OFF=off, N=Nn)))
            except Exception as exc:
                rec("err_flat_off%d" % off, repr(exc)[:400])


try:
    main()
except Exception:
    rec("fatal", traceback.format_exc())

p = os.path.join(adir(), "probe3.json")
with open(p, "w") as fh:
    json.dump(RES, fh, indent=2, default=str)
print("wrote", p, flush=True)
