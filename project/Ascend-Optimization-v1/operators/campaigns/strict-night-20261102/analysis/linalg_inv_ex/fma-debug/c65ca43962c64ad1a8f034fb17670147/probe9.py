"""Probe 9: which trailing-block formulation actually fuses on this backend?

For the 3x3 singular workload the required last pivot is exactly
-9.520052302036675e-09 (bit-confirmed against the reference LU).  On device the
flat sweep gives +5.208473652373868e-09, which is the "product rounded first"
variant; the reference needs the fused accumulate.  This probe runs the same
sweep with several formulations of the trailing update and reports the pivot.
"""
import itertools
import os

import numpy as np
import torch
import triton
import triton.language as tl

np.seterr(all="ignore")
np.set_printoptions(precision=10, linewidth=200)


@triton.jit
def _lu_probe(A, LU, PIVOTS, ONE, N: tl.constexpr, BLOCK_N: tl.constexpr, BLK: tl.constexpr,
              GMODE: tl.constexpr):
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < N) & (cols[None, :] < N)
    base = pid * N * N
    work = tl.load(A + base + cols[None, :] * N + rows[:, None], mask=mask, other=0.0).to(tl.float32)
    pivot_base = pid * N
    live = rows < N
    for j in tl.range(0, N):
        start = (j // BLK) * BLK
        pend = start + BLK
        col_j = tl.sum(tl.where(cols[None, :] == j, work, 0.0), axis=1)
        abs_col = tl.where((rows >= j) & live, tl.abs(col_j), -1.0)
        pivot_val = tl.max(abs_col, axis=0)
        pivot_row = tl.min(tl.where(abs_col == pivot_val, rows, BLOCK_N), axis=0)
        tl.store(PIVOTS + pivot_base + j, pivot_row)
        pivot = tl.sum(tl.where(rows == pivot_row, col_j, 0.0), axis=0)
        diag_j = tl.sum(tl.where(rows == j, col_j, 0.0), axis=0)
        row_j = tl.sum(tl.where(rows[:, None] == j, work, 0.0), axis=0)
        row_p = tl.sum(tl.where(rows[:, None] == pivot_row, work, 0.0), axis=0)
        work = tl.where((rows[:, None] == j) & (cols[None, :] < N), row_p[None, :], work)
        work = tl.where((rows[:, None] == pivot_row) & (cols[None, :] < N), row_j[None, :], work)
        rec = tl.where(pivot != 0.0, 1.0 / pivot, 1.0)
        work = tl.where((rows[:, None] > j) & (cols[None, :] == j) & live[:, None], work * rec, work)
        x_col = tl.where(rows == j, pivot, col_j)
        x_col = tl.where(rows == pivot_row, diag_j, x_col)
        x_col = tl.where(rows > j, x_col * rec, x_col)
        panel_in = ((rows[:, None] > j) & (cols[None, :] > j)
                    & (cols[None, :] < pend) & (cols[None, :] < N) & live[:, None])
        if GMODE == 0:
            work = tl.where(panel_in, x_col[:, None] * (-row_p[None, :]) + work, work)
        elif GMODE == 1:
            work = tl.where(panel_in, tl.math.fma(x_col[:, None], -row_p[None, :], work), work)
        else:
            one = tl.load(ONE)
            work = tl.where(panel_in, work + ((x_col[:, None] * (-row_p[None, :])) * one), work)
        u_after = row_p
        one2 = tl.load(ONE)
        for kk in tl.static_range(BLK - 1):
            k = start + kk
            kc = tl.minimum(k, N - 1)
            row_k = tl.sum(tl.where(rows[:, None] == kc, work, 0.0), axis=0)
            ljk = tl.sum(tl.where(cols == kc, row_p, 0.0), axis=0)
            ljk = tl.where(k < j, ljk, 0.0)
            u_after = u_after - (ljk * row_k) * one2
        work = tl.where((rows[:, None] == j) & (cols[None, :] >= pend) & (cols[None, :] < N),
                        u_after[None, :], work)
        gem = ((rows[:, None] >= pend) & (cols[None, :] >= pend)
               & (cols[None, :] < N) & live[:, None])
        if GMODE == 0:
            work = tl.where(gem, (-x_col[:, None]) * u_after[None, :] + work, work)
        elif GMODE == 1:
            work = tl.where(gem, tl.math.fma(-x_col[:, None], u_after[None, :], work), work)
        elif GMODE == 2:
            work = tl.where(gem, work - (x_col[:, None] * u_after[None, :]), work)
        elif GMODE == 3:
            one3 = tl.load(ONE)
            work = tl.where(gem, work - ((x_col[:, None] * u_after[None, :]) * one3), work)
        elif GMODE == 4:
            # accumulate into a separate register tile with a plain multiply-add
            tmp = x_col[:, None] * u_after[None, :]
            work = tl.where(gem, work - tmp, work)
    tl.store(LU + base + cols[None, :] * N + rows[:, None], work, mask=mask)


def build(shape, seed, singular, device="npu:0"):
    gen = torch.Generator().manual_seed(seed)

    def draw(shp):
        return torch.randn(shp, generator=gen).to(device=device, dtype=torch.float32)

    draw(shape)
    x = draw(shape)
    n = shape[-1]
    x = x + torch.eye(n, device=device, dtype=torch.float32) * (n + 1)
    if singular:
        x[..., -1, :] = x[..., 0, :]
    return x


def main():
    print("tl.math.fma:", hasattr(tl.math, "fma"), "tl.fma:", hasattr(tl, "fma"), flush=True)
    n = 3
    A = build([3, 3], 1730, True)
    A_dev = A.reshape(-1, n, n).contiguous()
    AT = A.cpu().reshape(-1, n, n).transpose(-1, -2).contiguous()
    lu_ref = torch.linalg.lu_factor_ex(AT)[0].numpy()
    print("torch LU(AT) last diag:", lu_ref[:, -1, -1], flush=True)
    print("full ref LU:\n", lu_ref[0], flush=True)
    for gmode in (0, 1, 3):
        for blk in (2, 3):
            lu = torch.empty(1, n, n, dtype=torch.float32, device="npu:0")
            piv = torch.empty(1, n, dtype=torch.int32, device="npu:0")
            unit = torch.ones(1, dtype=torch.float32, device="npu:0")
            try:
                _lu_probe[(1,)](A_dev, lu, piv, unit, N=n, BLOCK_N=16, BLK=blk, GMODE=gmode,
                                num_warps=4, num_stages=1)
                torch.npu.synchronize()
            except Exception as exc:  # noqa: BLE001
                print("  GMODE=%d BLK=%d failed: %s" % (gmode, blk, str(exc)[:300]), flush=True)
                continue
            got = lu.transpose(-1, -2).cpu().numpy()
            exact = np.array_equal(got.view(np.int32), lu_ref.view(np.int32))
            print("  GMODE=%d BLK=%d exact=%s lastdiag=%.10e  full=%s"
                  % (gmode, blk, exact, float(got[0, -1, -1]), np.array2string(got[0, -1], precision=5)), flush=True)


if __name__ == "__main__":
    main()
