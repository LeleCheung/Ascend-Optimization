"""Probe correct alternatives for value+index reduction on this backend."""
import json
import os

import torch
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
DEV = "npu"
BH, AH, BW, AW = 4, 2, 4, 2
BIG = 1 << 20


@triton.jit
def _combine(a_val, a_idx, b_val, b_idx):
    take_b = (b_val > a_val) | (b_val != b_val)
    return tl.where(take_b, b_val, a_val), tl.where(take_b, b_idx, a_idx)


def _load_x4():
    r = tl.arange(0, BH * AH)
    c = tl.arange(0, BW * AW)


@triton.jit
def k_tuple_f32_idx(x_ptr, v_ptr, i_ptr, BH: tl.constexpr, AH: tl.constexpr,
                    BW: tl.constexpr, AW: tl.constexpr):
    r = tl.arange(0, BH * AH)
    c = tl.arange(0, BW * AW)
    x = tl.load(x_ptr + r[:, None] * (BW * AW) + c[None, :])
    x4 = tl.reshape(x, (BH, AH, BW, AW))
    kv = tl.broadcast_to(tl.arange(0, AW)[None, None, None, :],
                         (BH, AH, BW, AW)).to(tl.float32)
    v, i = tl.reduce((x4, kv), 3, _combine)
    o = tl.arange(0, BH)[:, None, None] * (AH * BW) + tl.arange(0, AH)[None, :, None] * BW + tl.arange(0, BW)[None, None, :]
    tl.store(v_ptr + o, v)
    tl.store(i_ptr + o, i)


@triton.jit
def k_argmax(x_ptr, v_ptr, i_ptr, BH: tl.constexpr, AH: tl.constexpr,
             BW: tl.constexpr, AW: tl.constexpr):
    r = tl.arange(0, BH * AH)
    c = tl.arange(0, BW * AW)
    x = tl.load(x_ptr + r[:, None] * (BW * AW) + c[None, :])
    x4 = tl.reshape(x, (BH, AH, BW, AW))
    v = tl.max(x4, 3)
    i = tl.argmax(x4, 3)
    o = tl.arange(0, BH)[:, None, None] * (AH * BW) + tl.arange(0, AH)[None, :, None] * BW + tl.arange(0, BW)[None, None, :]
    tl.store(v_ptr + o, v)
    tl.store(i_ptr + o, i.to(tl.int32))


@triton.jit
def k_max_minpos(x_ptr, v_ptr, i_ptr, BH: tl.constexpr, AH: tl.constexpr,
                 BW: tl.constexpr, AW: tl.constexpr):
    r = tl.arange(0, BH * AH)
    c = tl.arange(0, BW * AW)
    x = tl.load(x_ptr + r[:, None] * (BW * AW) + c[None, :])
    x4 = tl.reshape(x, (BH, AH, BW, AW))
    lane = tl.broadcast_to(tl.arange(0, AW)[None, None, None, :],
                           (BH, AH, BW, AW))
    v = tl.max(x4, 3)
    i = tl.min(tl.where(x4 == v[:, :, :, None], lane, BIG), 3)
    o = tl.arange(0, BH)[:, None, None] * (AH * BW) + tl.arange(0, AH)[None, :, None] * BW + tl.arange(0, BW)[None, None, :]
    tl.store(v_ptr + o, v)
    tl.store(i_ptr + o, i)


@triton.jit
def k_permute(x_ptr, v_ptr, i_ptr, BH: tl.constexpr, AH: tl.constexpr,
              BW: tl.constexpr, AW: tl.constexpr):
    r = tl.arange(0, BH * AH)
    c = tl.arange(0, BW * AW)
    x = tl.load(x_ptr + r[:, None] * (BW * AW) + c[None, :])
    x4 = tl.reshape(x, (BH, AH, BW, AW))
    xp = tl.permute(x4, (0, 2, 1, 3))
    x3 = tl.reshape(xp, (BH, BW, AH * AW))
    lane = tl.broadcast_to(tl.arange(0, AH * AW)[None, None, :], (BH, BW, AH * AW))
    v = tl.max(x3, 2)
    i = tl.min(tl.where(x3 == v[:, :, None], lane, BIG), 2)
    o = tl.arange(0, BH)[:, None] * BW + tl.arange(0, BW)[None, :]
    tl.store(v_ptr + o, v)
    tl.store(i_ptr + o, i)


@triton.jit
def k_chain(x_ptr, v_ptr, i_ptr, BH: tl.constexpr, AH: tl.constexpr,
            BW: tl.constexpr, AW: tl.constexpr):
    """Two-step fold with max/min-where only: no tuple reduce, no permute."""
    r = tl.arange(0, BH * AH)
    c = tl.arange(0, BW * AW)
    x = tl.load(x_ptr + r[:, None] * (BW * AW) + c[None, :])
    x4 = tl.reshape(x, (BH, AH, BW, AW))
    kwl = tl.broadcast_to(tl.arange(0, AW)[None, None, None, :],
                          (BH, AH, BW, AW))
    v1 = tl.max(x4, 3)
    k1 = tl.min(tl.where(x4 == v1[:, :, :, None], kwl, BIG), 3)
    v2 = tl.max(v1, 1)
    row = tl.min(tl.where(v1 == v2[:, None, :], tl.arange(0, AH)[None, :, None], BIG), 1)
    kw = tl.min(tl.where(tl.arange(0, AH)[None, :, None] == row[:, None, :], k1, BIG), 1)
    o = tl.arange(0, BH)[:, None] * BW + tl.arange(0, BW)[None, :]
    tl.store(v_ptr + o, v2)
    tl.store(i_ptr + o, row * AW + kw)


def main():
    N = BH * AH
    x = torch.arange(N * N, dtype=torch.float32, device=DEV)
    xm = x.reshape(N, N)
    block = xm.reshape(BH, AH, BW, AW)
    exp_v1 = block.amax(dim=3)                      # [BH, AH, BW]
    exp_k1 = block.argmax(dim=3)                    # [BH, AH, BW]
    exp_v2 = block.amax(dim=3).amax(dim=1)          # [BH, BW]
    exp_k2 = block.permute(0, 2, 1, 3).reshape(BH, BW, AH * AW).argmax(dim=2)
    exp_k1f = block.permute(0, 2, 1, 3).reshape(BH, BW, AH * AW).amax(dim=2)

    res = {}

    def run(kern, name, flat_idx):
        v = torch.full((flat_idx.numel(),), -5.0, dtype=torch.float32, device=DEV)
        i = torch.full((flat_idx.numel(),), -9, dtype=torch.int32, device=DEV)
        kern[(1,)](x, v, i, BH, AH, BW, AW)
        res[name + "_val_ok"] = bool(torch.equal(v.reshape(flat_idx.shape), exp_v1))
        res[name + "_idx_ok"] = bool(torch.equal(i.reshape(flat_idx.shape), exp_k1.to(torch.int32)))
        res[name + "_got_i0"] = i[0:4].tolist()
        res[name + "_exp_i0"] = exp_k1.flatten()[0:4].tolist()

    exp_k1_shape = torch.empty(BH, AH, BW)
    run(k_tuple_f32_idx, "tuple_f32", exp_k1_shape)
    run(k_argmax, "argmax", exp_k1_shape)
    run(k_max_minpos, "max_minpos", exp_k1_shape)

    def run2(kern, name):
        v = torch.full((BH, BW), -5.0, dtype=torch.float32, device=DEV)
        i = torch.full((BH, BW), -9, dtype=torch.int32, device=DEV)
        kern[(1,)](x, v, i, BH, AH, BW, AW)
        res[name + "_val_ok"] = bool(torch.equal(v, exp_v2))
        res[name + "_idx_ok"] = bool(torch.equal(i, exp_k2.to(torch.int32)))
        res[name + "_got_i0"] = i[0].tolist()
        res[name + "_exp_i0"] = exp_k2[0].tolist()
        res[name + "_got_v0"] = v[0].tolist()
        res[name + "_exp_v0"] = exp_v2[0].tolist()

    run2(k_permute, "permute")
    run2(k_chain, "chain")

    with open(os.path.join(ART, "probe2.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res))


main()
