"""Localise the non-finite output and grid-search the LU arithmetic on the NPU.

Reports, for every authoritative workload:
  * tl.trans sanity (is transposition usable on this backend?)
  * candidate inverse vs torch reference (max_rel, bit-exact count), info
  * LU factors vs the numpy LAPACK emulation
Variants: multiplier scaling by rounded reciprocal (LAPACK dscal) vs per-element
division, both with index-based transposed access (no tl.trans).
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
def transk(X, Y, BLOCK: tl.constexpr):
    r = tl.arange(0, BLOCK)
    c = tl.arange(0, BLOCK)
    a = tl.load(X + r[:, None] * BLOCK + c[None, :])
    b = tl.trans(a)
    tl.store(Y + r[:, None] * BLOCK + c[None, :], b)


@triton.jit
def inv_kernel(
    A, INV, INFO, PIV, LUOUT,
    N: tl.constexpr, BLOCK_N: tl.constexpr, COL_TILE: tl.constexpr,
    RECIP: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_N)
    m = (rows[:, None] < N) & (cols[None, :] < N)
    # work[i, j] = A[j, i]  (transposed access)
    work = tl.load(A + pid * N * N + cols[None, :] * N + rows[:, None], mask=m, other=0.0)
    work = work.to(tl.float32)
    pbase = pid * N
    singular = 0

    for j in tl.range(0, N):
        col_j = tl.sum(tl.where(cols[None, :] == j, work, 0.0), axis=1)
        ac = tl.abs(col_j)
        ac = tl.where(rows >= j, ac, -1.0)
        ac = tl.where(rows < N, ac, -1.0)
        pv = tl.max(ac, axis=0)
        jp = tl.min(tl.where(ac == pv, rows, BLOCK_N), axis=0)
        tl.store(PIV + pbase + j, jp)
        pivot = tl.sum(tl.where(rows == jp, col_j, 0.0), axis=0)
        row_j = tl.sum(tl.where(rows[:, None] == j, work, 0.0), axis=0)
        row_p = tl.sum(tl.where(rows[:, None] == jp, work, 0.0), axis=0)
        work = tl.where((rows[:, None] == j) & (cols[None, :] < N), row_p[None, :], work)
        work = tl.where((rows[:, None] == jp) & (cols[None, :] < N), row_j[None, :], work)
        smask = (rows[:, None] > j) & (cols[None, :] == j)
        if RECIP:
            rec = tl.where(pivot != 0.0, 1.0 / pivot, 1.0)
            work = tl.where(smask, work * rec, work)
        else:
            den = tl.where(pivot != 0.0, pivot, 1.0)
            work = tl.where(smask, work / den, work)
        singular = tl.where((pivot == 0.0) & (singular == 0), j + 1, singular)
        x_col = tl.sum(tl.where(cols[None, :] == j, work, 0.0), axis=1)
        u_row = tl.sum(tl.where(rows[:, None] == j, work, 0.0), axis=0)
        upd = (rows[:, None] > j) & (cols[None, :] > j)
        work = tl.where(upd, work - x_col[:, None] * u_row[None, :], work)

    tl.store(LUOUT + pid * N * N + rows[:, None] * N + cols[None, :], work, mask=m)

    tc = tl.arange(0, COL_TILE)
    for ts in tl.range(0, N, COL_TILE):
        rhs = tl.where(rows[:, None] == (ts + tc[None, :]), 1.0, 0.0)
        for jp2 in tl.range(0, N):
            pr = tl.load(PIV + pbase + jp2)
            rj = tl.sum(tl.where(rows[:, None] == jp2, rhs, 0.0), axis=0)
            rp = tl.sum(tl.where(rows[:, None] == pr, rhs, 0.0), axis=0)
            rhs = tl.where(rows[:, None] == jp2, rp[None, :], rhs)
            rhs = tl.where(rows[:, None] == pr, rj[None, :], rhs)
        for jf in tl.range(0, N):
            lc = tl.sum(tl.where(cols[None, :] == jf, work, 0.0), axis=1)
            lc = tl.where(rows > jf, lc, 0.0)
            rb = tl.sum(tl.where(rows[:, None] == jf, rhs, 0.0), axis=0)
            rhs = rhs - lc[:, None] * rb[None, :]
        for jr in tl.range(0, N):
            jb = N - 1 - jr
            dc = tl.sum(tl.where(cols[None, :] == jb, work, 0.0), axis=1)
            dv = tl.sum(tl.where(rows == jb, dc, 0.0), axis=0)
            rb = tl.sum(tl.where(rows[:, None] == jb, rhs, 0.0), axis=0) / dv
            rhs = tl.where(rows[:, None] == jb, rb[None, :], rhs)
            uc = tl.sum(tl.where(cols[None, :] == jb, work, 0.0), axis=1)
            uc = tl.where(rows < jb, uc, 0.0)
            rhs = rhs - uc[:, None] * rb[None, :]
        off = pid * N * N + (ts + tc[None, :]) * N + rows[:, None]
        mk = (rows[:, None] < N) & ((ts + tc[None, :]) < N)
        tl.store(INV + off, rhs, mask=mk)

    tl.store(INFO + pid, singular)


def build(spec, seed):
    n = spec["shape"][-1]
    g = torch.Generator().manual_seed(seed)
    _ = torch.randn(spec["shape"], generator=g)
    x = torch.randn(spec["shape"], generator=g)
    x = x + torch.eye(n) * (n + 1)
    if spec.get("singular"):
        x[..., -1, :] = x[..., 0, :]
    return x.to(torch.float32).contiguous()


WL = [
    ("c0", {"shape": [3, 3], "singular": False}, 1729),
    ("c1", {"shape": [3, 3], "singular": True}, 1730),
    ("c2", {"shape": [2, 7, 7], "singular": False}, 1731),
    ("c3", {"shape": [2, 7, 7], "singular": True}, 1732),
    ("t0", {"shape": [16, 16]}, 1729),
    ("t1", {"shape": [8, 32, 32]}, 1730),
    ("t2", {"shape": [16, 64, 64]}, 1731),
]

report = {"trans_ok": None, "cases": {}}

# --- tl.trans sanity -----------------------------------------------------
try:
    xb = torch.arange(16, dtype=torch.float32, device="npu").reshape(4, 4).contiguous()
    yb = torch.empty_like(xb)
    transk[(1,)](xb, yb, BLOCK=4, num_warps=1)
    got = yb.cpu().numpy()
    report["trans_ok"] = bool(np.array_equal(got, xb.cpu().numpy().T))
    report["trans_got"] = got.tolist()
except Exception as exc:
    report["trans_ok"] = repr(exc)[:300]

# --- variant grid -------------------------------------------------------
for name, spec, seed in WL:
    n = spec["shape"][-1]
    A = build(spec, seed)
    flat = A.reshape(-1, n, n)
    batch = flat.shape[0]
    ref, ref_info = torch.linalg.inv_ex(A, check_errors=False)
    blk = max(triton.next_power_of_2(n), 4)
    ct = blk if blk <= 32 else 32
    entry = {"ref_info": ref_info.reshape(-1).tolist(),
             "ref_absmax": float(ref.abs().max()), "variants": {}}
    An = flat.to("npu")
    for label, recip in (("recip", True), ("div", False)):
        out = torch.empty(batch, n, n, dtype=torch.float32, device="npu")
        info = torch.empty(batch, dtype=torch.int32, device="npu")
        piv = torch.empty(batch, n, dtype=torch.int32, device="npu")
        lu = torch.empty(batch, n, n, dtype=torch.float32, device="npu")
        try:
            with torch_device_fn.device(An.device):
                inv_kernel[(batch,)](An, out, info, piv, lu, N=n, BLOCK_N=blk,
                                     COL_TILE=ct, RECIP=recip, num_warps=4)
            o = out.cpu()
            r = ref.cpu()
            exact = int((o.view(torch.int32) == r.view(torch.int32)).sum())
            d = (o.double() - r.double()).abs()
            rel = (d / r.double().abs().clamp_min(1e-300))
            finite = bool(torch.isfinite(o).all())
            entry["variants"][label] = {
                "info": info.cpu().reshape(-1).tolist(),
                "finite": finite,
                "exact": exact,
                "numel": int(o.numel()),
                "max_rel": float(rel.max()),
                "absmax": float(o.abs().max()),
                "pivots": piv.cpu().reshape(-1, n)[0].tolist(),
                "lu_diag0": lu.cpu().reshape(-1, n, n)[0].diagonal().tolist(),
            }
        except Exception as exc:
            entry["variants"][label] = {"error": repr(exc)[:300]}
    report["cases"][name] = entry

with open(os.path.join(OUT, "probe5.json"), "w") as f:
    json.dump(report, f, indent=1)
print(json.dumps(report, indent=1)[:9000])
