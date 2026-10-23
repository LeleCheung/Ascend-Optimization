"""Probe 2: separate memory-pattern costs and test a flat-contiguous 2x kernel.

Diagnostic only.
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


# ---------------------------------------------------------------- kernels
@triton.jit
def _k_flat(ptr_o, ptr_i, N: tl.constexpr):
    pid = tl.program_id(0)
    m = pid * N + tl.arange(0, N)
    tl.store(ptr_o + m, tl.load(ptr_i + m))


@triton.jit
def _k_aff2d(ptr_o, ptr_i, W: tl.constexpr, R: tl.constexpr):
    pid = tl.program_id(0)
    o = pid * (R * W) + (tl.arange(0, R) * W)[:, None] + tl.arange(0, W)[None, :]
    tl.store(ptr_o + o, tl.load(ptr_i + o))


@triton.jit
def _k_dyn2d(ptr_o, ptr_i, s_h, W: tl.constexpr, R: tl.constexpr, IH: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * R + tl.arange(0, R)
    c = tl.arange(0, W)
    src = ((rows % IH) * s_h)[:, None] + c[None, :]
    dst = (rows * W)[:, None] + c[None, :]
    tl.store(ptr_o + dst, tl.load(ptr_i + src))


@triton.jit
def _k_strst(ptr_o, ptr_i, W: tl.constexpr, R: tl.constexpr):
    pid = tl.program_id(0)
    r = tl.arange(0, R)
    c = tl.arange(0, W)
    src = pid * (R * W) + (r * W)[:, None] + c[None, :]
    dst = pid * (R * 2 * W) + (r * 2 * W)[:, None] + c[None, :]
    tl.store(ptr_o + dst, tl.load(ptr_i + src))


@triton.jit
def _k_new(ptr_o, ptr_i, total_in, qper,
           R: tl.constexpr, W: tl.constexpr, IH: tl.constexpr,
           W2: tl.constexpr, IMG_IN: tl.constexpr, IMG_OUT: tl.constexpr):
    pid = tl.program_id(0)
    nc = pid // qper
    qb = pid % qper
    q0 = qb * R
    N: tl.constexpr = R * W
    m = tl.arange(0, N)
    inb = nc * IMG_IN
    a0 = inb + (q0 - 1) * W

    fa = a0 + m
    keep = fa >= 0
    P = tl.load(ptr_i + fa, mask=keep, other=0.0).to(tl.float32)
    Q = tl.load(ptr_i + (a0 + W + m)).to(tl.float32)
    fb = a0 + 2 * W + m
    S = tl.load(ptr_i + fb, mask=fb < total_in, other=0.0).to(tl.float32)

    top = (m < W) & (q0 == 0)
    P = tl.where(top, Q, P)
    bot = (m >= N - W) & (qb == qper - 1)
    S = tl.where(bot, Q, S)

    ve = 0.25 * P + 0.75 * Q
    vo = 0.75 * Q + 0.25 * S

    col = m % W
    ip = tl.where(col == 0, m, m - 1)
    inx = tl.where(col == W - 1, m, m + 1)

    ee = 0.25 * tl.gather(ve, ip, 0) + 0.75 * ve
    eo = 0.75 * ve + 0.25 * tl.gather(ve, inx, 0)
    oe = 0.25 * tl.gather(vo, ip, 0) + 0.75 * vo
    oo = 0.75 * vo + 0.25 * tl.gather(vo, inx, 0)

    te = tl.reshape(tl.reshape(tl.join(ee, eo), (N, 2)), (R, W2))
    to = tl.reshape(tl.reshape(tl.join(oe, oo), (N, 2)), (R, W2))

    r = tl.arange(0, R)
    c = tl.arange(0, W2)
    ob = nc * IMG_OUT + (2 * q0) * W2
    off = ob + (r * 2 * W2)[:, None] + c[None, :]
    tl.store(ptr_o + off, te.to(ptr_o.dtype.element_ty))
    tl.store(ptr_o + off + W2, to.to(ptr_o.dtype.element_ty))


# ---------------------------------------------------------------- timing
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
SMALL = (1, 16, 64, 64)
SH, SW = 128, 128


def ref_op(x, oh, ow):
    return torch.ops.aten.upsample_bilinear2d(x, [oh, ow], False, None, None)


def probe_floors():
    n = 1 << 23  # 8.4M elements, the big-case output volume
    src = torch.randn(n, device=DEV, dtype=torch.float16)
    dst = torch.empty(n, device=DEV, dtype=torch.float16)
    for R, W in ((16, 128), (32, 128)):
        N = R * W
        g = (n // N,)
        try:
            rec("wall_flat_R%d" % R, time_wall(lambda g=g, N=N: _k_flat[g](dst, src, N=N)))
        except Exception as exc:
            rec("err_flat_R%d" % R, repr(exc))
        try:
            rec("wall_aff2d_R%d" % R,
                time_wall(lambda g=g, W=W, R=R: _k_aff2d[g](dst, src, W=W, R=R)))
        except Exception as exc:
            rec("err_aff2d_R%d" % R, repr(exc))
        try:
            rec("wall_dyn2d_R%d" % R,
                time_wall(lambda g=g, W=W, R=R: _k_dyn2d[g](dst, src, W, W=W, R=R, IH=128)))
        except Exception as exc:
            rec("err_dyn2d_R%d" % R, repr(exc))
        try:
            rec("wall_strst_R%d" % R,
                time_wall(lambda g=g, W=W, R=R: _k_strst[g](dst, src, W=W, R=R)))
        except Exception as exc:
            rec("err_strst_R%d" % R, repr(exc))


def run_new(x, OH, OW, R, tag, iters):
    N, C, IH, IW = x.shape
    nimg = N * C
    qper = IH // R
    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    grid = (nimg * qper,)
    W2 = 2 * IW
    img_in = IH * IW
    img_out = OH * OW
    total_in = x.numel()

    def call():
        _k_new[grid](out, x, total_in, qper, R=R, W=IW, IH=IH, W2=W2,
                     IMG_IN=img_in, IMG_OUT=img_out)

    call()
    torch.npu.synchronize()
    rec("wall_new_%s" % tag, time_wall(call, iters=iters))
    ref = ref_op(x, OH, OW)
    d = (out.float() - ref.float()).abs().max().item()
    rec("maxdiff_new_%s" % tag, d)


def main():
    x16 = torch.randn(BIG, device=DEV, dtype=torch.float16)
    x32 = torch.randn(BIG, device=DEV, dtype=torch.float32)
    s16 = torch.randn(SMALL, device=DEV, dtype=torch.float16)
    probe_floors()
    for R in (8, 16):
        try:
            run_new(x16, BH, BW, R, "big16_R%d" % R, 10)
        except Exception as exc:
            rec("err_new_big16_R%d" % R, repr(exc))
    try:
        run_new(x32, BH, BW, 16, "big32_R16", 10)
    except Exception as exc:
        rec("err_new_big32_R16", repr(exc))
    for R in (8, 16):
        try:
            run_new(s16, SH, SW, R, "small16_R%d" % R, 50)
        except Exception as exc:
            rec("err_new_small16_R%d" % R, repr(exc))


try:
    main()
except Exception:
    rec("fatal", traceback.format_exc())

p = os.path.join(adir(), "probe2.json")
with open(p, "w") as fh:
    json.dump(RES, fh, indent=2, default=str)
print("wrote", p, flush=True)
