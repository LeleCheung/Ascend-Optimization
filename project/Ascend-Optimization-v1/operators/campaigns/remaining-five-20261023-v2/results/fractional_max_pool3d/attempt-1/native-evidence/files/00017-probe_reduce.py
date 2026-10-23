"""Isolate which primitive of the integer-ratio kernel is wrong.

Single tiny case: 1x1x8x8x8 fp32, kernel 2x2x2 -> out 4x4x4, input = arange
(unique, increasing) so every expected value/index is known analytically.
"""
import json
import os
import sys

import torch
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
DEV = "npu"
BH, AH, BW, AW = 4, 2, 4, 2


@triton.jit
def _combine(a_val, a_idx, b_val, b_idx):
    take_b = (b_val > a_val) | (b_val != b_val)
    return tl.where(take_b, b_val, a_val), tl.where(take_b, b_idx, a_idx)


@triton.jit
def k_raw(x_ptr, o_ptr, N: tl.constexpr):
    r = tl.arange(0, N)
    x = tl.load(x_ptr + r[:, None] * N + r[None, :])
    tl.store(o_ptr + r[:, None] * N + r[None, :], x)


@triton.jit
def k_roundtrip(x_ptr, o_ptr, N: tl.constexpr, BH: tl.constexpr, AH: tl.constexpr,
                BW: tl.constexpr, AW: tl.constexpr):
    r = tl.arange(0, N)
    x = tl.load(x_ptr + r[:, None] * N + r[None, :])
    x4 = tl.reshape(x, (BH, AH, BW, AW))
    x2 = tl.reshape(x4, (N, N))
    tl.store(o_ptr + r[:, None] * N + r[None, :], x2)


@triton.jit
def k_red_max(x_ptr, o_ptr, N: tl.constexpr, BH: tl.constexpr, AH: tl.constexpr,
              BW: tl.constexpr, AW: tl.constexpr):
    r = tl.arange(0, N)
    x = tl.load(x_ptr + r[:, None] * N + r[None, :])
    x4 = tl.reshape(x, (BH, AH, BW, AW))
    v = tl.max(x4, 3)
    o = tl.arange(0, BH)[:, None, None] * (AH * BW) + tl.arange(0, AH)[None, :, None] * BW + tl.arange(0, BW)[None, None, :]
    tl.store(o_ptr + o, v)


@triton.jit
def k_red_tuple(x_ptr, o_ptr, x_ptr2, N: tl.constexpr, BH: tl.constexpr, AH: tl.constexpr,
                BW: tl.constexpr, AW: tl.constexpr):
    r = tl.arange(0, N)
    x = tl.load(x_ptr + r[:, None] * N + r[None, :])
    x4 = tl.reshape(x, (BH, AH, BW, AW))
    kv = tl.broadcast_to(tl.arange(0, AW)[None, None, None, :],
                         (BH, AH, BW, AW)).to(tl.int32)
    v1, k1 = tl.reduce((x4, kv), 3, _combine)
    o = tl.arange(0, BH)[:, None, None] * (AH * BW) + tl.arange(0, AH)[None, :, None] * BW + tl.arange(0, BW)[None, None, :]
    tl.store(o_ptr + o, v1)
    tl.store(x_ptr2 + o, k1)


@triton.jit
def k_full_2d(x_ptr, o_ptr, i_ptr, BH: tl.constexpr, AH: tl.constexpr,
              BW: tl.constexpr, AW: tl.constexpr):
    """Full (h,w) fold of the 8x8 block with the 4D tile, then index decode."""
    r = tl.arange(0, BH * AH)
    c = tl.arange(0, BW * AW)
    x = tl.load(x_ptr + r[:, None] * (BW * AW) + c[None, :])
    x4 = tl.reshape(x, (BH, AH, BW, AW))
    kv = tl.broadcast_to(tl.arange(0, AW)[None, None, None, :],
                         (BH, AH, BW, AW)).to(tl.int32)
    khv = tl.broadcast_to(tl.arange(0, AH)[None, :, None],
                          (BH, AH, BW)).to(tl.int32)
    v1, k1 = tl.reduce((x4, kv), 3, _combine)
    v2, k2 = tl.reduce((v1, khv * AW + k1), 1, _combine)
    o = tl.arange(0, BH)[:, None] * BW + tl.arange(0, BW)[None, :]
    tl.store(o_ptr + o, v2)
    tl.store(i_ptr + o, k2)


@triton.jit
def k_direct4d(x_ptr, o_ptr, i_ptr, BH: tl.constexpr, AH: tl.constexpr,
               BW: tl.constexpr, AW: tl.constexpr):
    """Same fold but the 4D tile is loaded directly (no reshape)."""
    off = (tl.arange(0, BH)[:, None, None, None] * AH + tl.arange(0, AH)[None, :, None, None]) * (BW * AW) \
        + (tl.arange(0, BW)[None, None, :, None] * AW + tl.arange(0, AW)[None, None, None, :])
    x4 = tl.load(x_ptr + off)
    kv = tl.broadcast_to(tl.arange(0, AW)[None, None, None, :],
                         (BH, AH, BW, AW)).to(tl.int32)
    khv = tl.broadcast_to(tl.arange(0, AH)[None, :, None],
                          (BH, AH, BW)).to(tl.int32)
    v1, k1 = tl.reduce((x4, kv), 3, _combine)
    v2, k2 = tl.reduce((v1, khv * AW + k1), 1, _combine)
    o = tl.arange(0, BH)[:, None] * BW + tl.arange(0, BW)[None, :]
    tl.store(o_ptr + o, v2)
    tl.store(i_ptr + o, k2)


def main():
    N = BH * AH
    x = torch.arange(N * N, dtype=torch.float32, device=DEV)
    res = {}

    def alloc(shape):
        return torch.full(shape, -7.0, dtype=torch.float32, device=DEV)

    raw = alloc((N, N))
    k_raw[(1,)](x, raw, N)
    res["raw_load_ok"] = bool(torch.equal(raw, x.reshape(N, N)))

    rt = alloc((N, N))
    k_roundtrip[(1,)](x, rt, N, BH, AH, BW, AW)
    res["roundtrip_ok"] = bool(torch.equal(rt, x.reshape(N, N)))

    xm = x.reshape(N, N)
    # expected max over the (AH, AW) sub-block, per (BH, BW) cell
    exp_max = xm.reshape(BH, AH, BW, AW).amax(dim=3).reshape(BH * AH, BW)
    got = alloc((BH * AH, BW))
    k_red_max[(1,)](x, got, N, BH, AH, BW, AW)
    res["reduce_max_ok"] = bool(torch.equal(got, exp_max))
    res["reduce_max_got0"] = got[0].tolist()
    res["reduce_max_exp0"] = exp_max[0].tolist()

    gv = alloc((BH * AH, BW))
    gi = torch.full((BH * AH, BW), -9, dtype=torch.int32, device=DEV)
    k_red_tuple[(1,)](x, gv, gi, N, BH, AH, BW, AW)
    exp_k_full = xm.reshape(BH, AH, BW, AW).argmax(dim=3)  # [BH, AH, BW]
    res["reduce_tuple_val_ok"] = bool(torch.equal(gv, exp_max))
    res["reduce_tuple_idx_ok"] = bool(torch.equal(gi, exp_k_full))

    # full fold: max over (kh, kw) -> 4x4 values, and k2 = kh*AW + kw
    block = xm.reshape(BH, AH, BW, AW)
    exp_v2 = block.amax(dim=3).amax(dim=1)  # [BH, BW]
    exp_flat = block.permute(0, 2, 1, 3).reshape(BH, BW, AH * AW)
    exp_k2 = exp_flat.argmax(dim=2)

    fv = alloc((BH, BW))
    fi = torch.full((BH, BW), -9, dtype=torch.int32, device=DEV)
    k_full_2d[(1,)](x, fv, fi, BH, AH, BW, AW)
    res["full_val_ok"] = bool(torch.equal(fv, exp_v2))
    res["full_idx_ok"] = bool(torch.equal(fi, exp_k2))
    res["full_got_v0"] = fv[0].tolist()
    res["full_exp_v0"] = exp_v2[0].tolist()
    res["full_got_i0"] = fi[0].tolist()
    res["full_exp_i0"] = exp_k2[0].tolist()

    dv = alloc((BH, BW))
    di = torch.full((BH, BW), -9, dtype=torch.int32, device=DEV)
    k_direct4d[(1,)](x, dv, di, BH, AH, BW, AW)
    res["direct4d_val_ok"] = bool(torch.equal(dv, exp_v2))
    res["direct4d_idx_ok"] = bool(torch.equal(di, exp_k2))

    with open(os.path.join(ART, "probe_reduce.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res))


main()
