"""One tiny kernel: verify transposed access patterns and FMA contraction.

Checks (single compilation, BLOCK=4):
  1. strided transposed load (offsets c*B + r)  -> returns X^T ?
  2. tl.trans(X)                                -> returns X^T ?
  3. strided transposed store                   -> lands as Y[t*B + r] ?
  4. `a - m * u` : does the backend contract to FMA (single rounding)?
"""

import json
import os

import numpy as np
import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn

OUT = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")


@triton.jit
def probe(X, Y, TL, TS, FA, FB, FC, FRES,
          B: tl.constexpr):
    r = tl.arange(0, B)
    c = tl.arange(0, B)
    # 1. strided transposed load: value(i,j) = X[j, i]
    st = tl.load(X + c[None, :] * B + r[:, None])
    tl.store(TL + r[:, None] * B + c[None, :], st)
    # 2. tl.trans of a contiguous load
    a = tl.load(X + r[:, None] * B + c[None, :])
    tr = tl.trans(a)
    tl.store(TS + r[:, None] * B + c[None, :], tr)
    # 3. strided transposed store: writes val(i, t) to Y[t*B + i]
    tl.store(Y + c[None, :] * B + r[:, None], a)
    # 4. FMA contraction probe
    fa = tl.load(FA)
    fb = tl.load(FB)
    fc = tl.load(FC)
    res = fa - fb * fc
    tl.store(FRES, res.to(tl.int32, bitcast=True))


B = 4
x = torch.arange(B * B, dtype=torch.float32, device="npu").reshape(B, B).contiguous()
y = torch.full((B, B), -1.0, dtype=torch.float32, device="npu")
t1 = torch.full((B, B), -1.0, dtype=torch.float32, device="npu")
t2 = torch.full((B, B), -1.0, dtype=torch.float32, device="npu")
# values chosen so fma (single rounding) and mul+sub (two roundings) differ
fa = torch.tensor([1.0], dtype=torch.float32, device="npu")
fb = torch.tensor([1.00000011920928955], dtype=torch.float32, device="npu")
fc = torch.tensor([1.00000011920928955], dtype=torch.float32, device="npu")
fres = torch.zeros(1, dtype=torch.int32, device="npu")

out = {}
try:
    with torch_device_fn.device(x.device):
        probe[(1,)](x, y, t1, t2, fa, fb, fc, fres, B=B, num_warps=1)
    xn = x.cpu().numpy()
    out["strided_load_is_transpose"] = bool(np.array_equal(t1.cpu().numpy(), xn.T))
    out["tl_trans_is_transpose"] = bool(np.array_equal(t2.cpu().numpy(), xn.T))
    # store wrote a(i,t) to Y[t*B+i]  =>  Y == xn.T
    out["strided_store_is_transpose"] = bool(np.array_equal(y.cpu().numpy(), xn.T))
    out["strided_store_got"] = y.cpu().numpy().tolist()
    a = np.float32(1.0)
    m = np.float32(1.00000011920928955)
    fma_bits = np.float32(np.float64(a) - np.float64(m) * np.float64(m))
    mul_bits = np.float32(a - np.float32(m * m))
    got = int(fres.cpu().numpy()[0])
    got_f = np.int32(got).view(np.float32) if got >= 0 else np.uint32(got).view(np.float32)
    out["fma_probe"] = {
        "kernel_value": float(got_f),
        "fma_value": float(fma_bits),
        "mulsubl_value": float(mul_bits),
        "matches_fma": float(got_f) == float(fma_bits),
        "matches_mulsub": float(got_f) == float(mul_bits),
    }
except Exception as exc:
    out["error"] = repr(exc)[:1500]

with open(os.path.join(OUT, "probe6.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1)[:4000])
