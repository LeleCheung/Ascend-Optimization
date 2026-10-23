"""Which element of the 4D-tile fold compiles at which tile size and dtype."""
import json
import os
import traceback

import torch
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
DEV = "npu"
BIG = tl.constexpr(1 << 24)


@triton.jit
def _combine(a_val, a_idx, b_val, b_idx):
    take_b = (b_val > a_val) | (b_val != b_val)
    return tl.where(take_b, b_val, a_val), tl.where(take_b, b_idx, a_idx)


@triton.jit
def k_vals(x_ptr, v_ptr, BH: tl.constexpr, AH: tl.constexpr, BW: tl.constexpr,
           AW: tl.constexpr):
    r = tl.arange(0, BH * AH)
    c = tl.arange(0, BW * AW)
    x = tl.load(x_ptr + r[:, None] * (BW * AW) + c[None, :])
    x4 = tl.reshape(x, (BH, AH, BW, AW))
    v1 = tl.max(x4, 3)
    v2 = tl.max(v1, 1)
    o = tl.arange(0, BH)[:, None] * BW + tl.arange(0, BW)[None, :]
    tl.store(v_ptr + o, v2)


@triton.jit
def k_chain4d(x_ptr, v_ptr, i_ptr, BH: tl.constexpr, AH: tl.constexpr,
              BW: tl.constexpr, AW: tl.constexpr):
    r = tl.arange(0, BH * AH)
    c = tl.arange(0, BW * AW)
    x = tl.load(x_ptr + r[:, None] * (BW * AW) + c[None, :])
    x4 = tl.reshape(x, (BH, AH, BW, AW))
    kwl = tl.broadcast_to(tl.arange(0, AW)[None, None, None, :], (BH, AH, BW, AW))
    khl = tl.broadcast_to(tl.arange(0, AH)[None, :, None], (BH, AH, BW))
    v1 = tl.max(x4, 3)
    k1 = tl.min(tl.where(x4 == v1[:, :, :, None], kwl, BIG), 3)
    v2 = tl.max(v1, 1)
    row = tl.min(tl.where(v1 == v2[:, None, :], khl, BIG), 1)
    kw = tl.min(tl.where(khl == row[:, None, :], k1, BIG), 1)
    o = tl.arange(0, BH)[:, None] * BW + tl.arange(0, BW)[None, :]
    tl.store(v_ptr + o, v2)
    tl.store(i_ptr + o, row * AW + kw)


@triton.jit
def k_nan(x_ptr, v_ptr, i_ptr, BH: tl.constexpr, AH: tl.constexpr,
          BW: tl.constexpr, AW: tl.constexpr):
    r = tl.arange(0, BH * AH)
    c = tl.arange(0, BW * AW)
    x = tl.load(x_ptr + r[:, None] * (BW * AW) + c[None, :])
    x4 = tl.reshape(x, (BH, AH, BW, AW))
    kwl = tl.broadcast_to(tl.arange(0, AW)[None, None, None, :], (BH, AH, BW, AW))
    khl = tl.broadcast_to(tl.arange(0, AH)[None, :, None], (BH, AH, BW))
    neg_inf = float("-inf")
    isn = x4 != x4
    fin = tl.where(isn, neg_inf, x4)
    v1 = tl.max(fin, 3)
    k1 = tl.min(tl.where(fin == v1[:, :, :, None], kwl, BIG), 3)
    ln1 = tl.max(tl.where(isn, kwl, -1), 3)
    v2 = tl.max(v1, 1)
    row = tl.min(tl.where(v1 == v2[:, None, :], khl, BIG), 1)
    kw = tl.min(tl.where(khl == row[:, None, :], k1, BIG), 1)
    nrow = tl.max(tl.where(ln1 >= 0, khl, -1), 1)
    nkw = tl.max(tl.where(khl == nrow[:, None, :], ln1, -1), 1)
    anyn = nrow >= 0
    o = tl.arange(0, BH)[:, None] * BW + tl.arange(0, BW)[None, :]
    tl.store(v_ptr + o, tl.where(anyn, tl.full((BH, BW), float("nan"), x_ptr.type.element_ty), v2))
    tl.store(i_ptr + o, tl.where(anyn, nrow * AW + nkw, row * AW + kw))


def expected(x, BH, AH, BW, AW):
    block = x.reshape(BH, AH, BW, AW)
    v2 = block.amax(dim=3).amax(dim=1)
    flat = block.permute(0, 2, 1, 3).reshape(BH, BW, AH * AW)
    return v2, flat.argmax(dim=2)


def main():
    res = {}
    for BH, BW in ((8, 8), (16, 16)):
        for dt in (torch.float32, torch.float16):
            AH = AW = 2
            n = BH * AH * BW * AW
            x = torch.arange(n, dtype=torch.float32, device=DEV).reshape(
                BH * AH, BW * AW).to(dt)
            exp_v, exp_i = expected(x.float(), BH, AH, BW, AW)
            key = f"{str(dt).split('.')[-1]}_{BH}x{BW}"
            for name, kern in (("vals", k_vals), ("chain4d", k_chain4d), ("nan", k_nan)):
                try:
                    v = torch.full((BH * BW,), -5.0, dtype=dt, device=DEV)
                    i = torch.full((BH * BW,), -9, dtype=torch.int32, device=DEV)
                    if name == "vals":
                        kern[(1,)](x, v, BH, AH, BW, AW)
                        res[f"{key}_{name}"] = {
                            "compile": "ok",
                            "val_ok": bool(torch.equal(v.reshape(BH, BW).float(), exp_v)),
                        }
                    else:
                        kern[(1,)](x, v, i, BH, AH, BW, AW)
                        res[f"{key}_{name}"] = {
                            "compile": "ok",
                            "val_ok": bool(torch.equal(v.reshape(BH, BW).float(), exp_v)),
                            "idx_ok": bool(torch.equal(i.reshape(BH, BW), exp_i.to(torch.int32))),
                        }
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc)
                    short = msg.splitlines()[-1][:180] if msg else type(exc).__name__
                    res[f"{key}_{name}"] = {"compile": "fail", "err": short,
                                            "ub": "ub overflow" in msg,
                                            "align": "cannot align 2 axis" in msg}
    with open(os.path.join(ART, "probe3.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res))


main()
