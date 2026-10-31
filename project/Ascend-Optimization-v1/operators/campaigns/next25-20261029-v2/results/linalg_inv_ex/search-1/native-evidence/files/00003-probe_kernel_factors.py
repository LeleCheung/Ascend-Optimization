"""Diagnostic: run a copy of the candidate kernel that also stores its LU factors,
compare them bit-wise with torch's lu_factor_ex(A^T), and report the kernel's
inverse output for the four correctness shapes (n=3 and n=7, regular/singular).
"""
import json
import os
import warnings

import numpy as np
import torch
import triton
import triton.language as tl

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")


@triton.jit
def _kern(A, INV, INFO, PIVOTS, FACT, N: tl.constexpr, BLOCK_N: tl.constexpr,
          COL_TILE: tl.constexpr):
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < N) & (cols[None, :] < N)
    offsets = pid * N * N + cols[None, :] * N + rows[:, None]
    work = tl.load(A + offsets, mask=mask, other=0.0).to(tl.float32)

    pivot_base = pid * N
    singular = 0
    col_mask = cols[None, :] < N

    for j_ind in tl.range(0, N):
        col_j = tl.sum(tl.where(cols[None, :] == j_ind, work, 0.0), axis=1)
        abs_col = tl.abs(col_j)
        abs_col = tl.where(rows < j_ind, -1.0, abs_col)
        abs_col = tl.where(rows < N, abs_col, -1.0)
        pivot_val = tl.max(abs_col, axis=0)
        pivot_row = tl.min(tl.where(abs_col == pivot_val, rows, BLOCK_N), axis=0)
        tl.store(PIVOTS + pivot_base + j_ind, pivot_row)
        row_j = tl.sum(tl.where(rows[:, None] == j_ind, work, 0.0), axis=0)
        row_p = tl.sum(tl.where(rows[:, None] == pivot_row, work, 0.0), axis=0)
        work = tl.where((rows[:, None] == j_ind) & col_mask, row_p, work)
        work = tl.where((rows[:, None] == pivot_row) & col_mask, row_j, work)
        pivot = tl.sum(tl.sum(tl.where((rows[:, None] == j_ind) & (cols[None, :] == j_ind),
                                       work, 0.0), axis=0), axis=0)
        singular = tl.where((pivot == 0.0) & (singular == 0), j_ind + 1, singular)
        col_vals = tl.sum(tl.where(cols[None, :] == j_ind, work, 0.0), axis=1)
        rcp = 1.0 / pivot
        multipliers = tl.where(rows > j_ind, col_vals * rcp, 0.0)
        work = tl.where((rows[:, None] > j_ind) & (cols[None, :] == j_ind),
                        multipliers[:, None], work)
        u_row = tl.sum(tl.where(rows[:, None] == j_ind, work, 0.0), axis=0)
        update_mask = (rows[:, None] > j_ind) & (cols[None, :] > j_ind)
        prod = multipliers[:, None] * u_row[None, :]
        work = tl.where(update_mask, work - prod, work)

    fact_off = pid * BLOCK_N * BLOCK_N + rows[:, None] * BLOCK_N + cols[None, :]
    tl.store(FACT + fact_off, work)

    tile_cols = tl.arange(0, COL_TILE)
    for tile_start in tl.range(0, N, COL_TILE):
        rhs = tl.where(rows[:, None] == (tile_start + tile_cols[None, :]), 1.0, 0.0)
        tile_mask = tile_cols[None, :] < COL_TILE
        for j_perm in tl.range(0, N):
            piv_row = tl.load(PIVOTS + pivot_base + j_perm)
            rhs_row_j = tl.sum(tl.where(rows[:, None] == j_perm, rhs, 0.0), axis=0)
            rhs_row_p = tl.sum(tl.where(rows[:, None] == piv_row, rhs, 0.0), axis=0)
            rhs = tl.where((rows[:, None] == j_perm) & tile_mask, rhs_row_p[None, :], rhs)
            rhs = tl.where((rows[:, None] == piv_row) & tile_mask, rhs_row_j[None, :], rhs)
        for j_fwd in tl.range(0, N):
            diag_val = tl.sum(tl.sum(tl.where((rows[:, None] == j_fwd) & (cols[None, :] == j_fwd),
                                              work, 0.0), axis=0), axis=0)
            rhs_row_j = tl.sum(tl.where(rows[:, None] == j_fwd, rhs, 0.0), axis=0)
            scaled_row = rhs_row_j / diag_val
            rhs = tl.where((rows[:, None] == j_fwd) & tile_mask, scaled_row[None, :], rhs)
            u_row = tl.sum(tl.where(rows[:, None] == j_fwd, work, 0.0), axis=0)
            u_row = tl.where(rows > j_fwd, u_row, 0.0)
            rhs = rhs - u_row[:, None] * scaled_row[None, :]
        for j_rev in tl.range(0, N):
            j_back = N - 1 - j_rev
            z_row = tl.sum(tl.where(rows[:, None] == j_back, rhs, 0.0), axis=0)
            l_row = tl.sum(tl.where(rows[:, None] == j_back, work, 0.0), axis=0)
            l_row = tl.where(cols < j_back, l_row, 0.0)
            rhs = rhs - l_row[:, None] * z_row[None, :]
        out_offsets = pid * N * N + rows[:, None] * N + (tile_start + tile_cols[None, :])
        out_mask = (rows[:, None] < N) & ((tile_start + tile_cols[None, :]) < N)
        tl.store(INV + out_offsets, rhs, mask=out_mask)
    tl.store(INFO + pid, singular)


def gen_inputs(spec, seed):
    dtype = getattr(torch, spec["dtype"])
    gen = torch.Generator().manual_seed(seed)

    def random(shape):
        return torch.randn(shape, generator=gen).to(dtype=dtype)

    random(spec["shape"])
    x = random(spec["shape"])
    x = x + torch.eye(x.shape[-1], dtype=dtype) * (x.shape[-1] + 1)
    if spec.get("singular"):
        x[..., -1, :] = x[..., 0, :]
    return x


cases = [
    ("c0_reg3", {"dtype": "float32", "shape": [3, 3], "singular": False}, 1729),
    ("c1_sing3", {"dtype": "float32", "shape": [3, 3], "singular": True}, 1730),
    ("c2_reg7", {"dtype": "float32", "shape": [2, 7, 7], "singular": False}, 1731),
    ("c3_sing7", {"dtype": "float32", "shape": [2, 7, 7], "singular": True}, 1732),
]

lines, out = [], {"cases": {}}
for name, spec, seed in cases:
    A = gen_inputs(spec, seed)
    n = spec["shape"][-1]
    Af = A.reshape(-1, n, n)
    bs = Af.shape[0]
    bn = max(triton.next_power_of_2(n), 4)
    ct = bn if bn <= 32 else 32
    inv = torch.empty(bs, n, n, dtype=torch.float32)
    info = torch.empty(bs, dtype=torch.int32)
    piv = torch.empty(bs, n, dtype=torch.int32)
    fact = torch.empty(bs, bn, bn, dtype=torch.float32)
    _kern[(bs,)](Af, inv, info, piv, fact, N=n, BLOCK_N=bn, COL_TILE=ct, num_warps=4)
    ref = torch.linalg.inv_ex(A, check_errors=False)
    rec = {"n": n, "BLOCK_N": bn, "COL_TILE": ct, "ref_info": ref.info.tolist(),
           "kernel_info": info.tolist(), "batches": []}
    lines.append(f"{name}: n={n} BLOCK_N={bn} kernel_info={info.tolist()} ref_info={ref.info.tolist()}")
    for b in range(bs):
        kf = fact[b].numpy()
        LUt = torch.linalg.lu_factor_ex(torch.from_numpy(np.ascontiguousarray(Af[b].numpy().T)))[0].numpy()
        kf_n = kf[:n, :n]
        exact = np.array_equal(kf_n, LUt)
        diff = np.abs(kf_n - LUt)
        bad = int((diff > 0).sum())
        idx = np.unravel_index(np.argmax(diff), diff.shape) if diff.size else (0, 0)
        kraw = inv[b].numpy()
        rin = ref.inverse.reshape(-1, n, n)[b].numpy()
        with np.errstate(invalid="ignore"):
            ok = np.isclose(rin, kraw, rtol=1e-4, atol=1e-4, equal_nan=True)
            rel = np.abs(kraw - rin) / np.maximum(np.abs(rin), np.float32(1e-30))
        e = {"b": b, "factor_bit_exact": bool(exact), "n_diff_entries": bad,
             "first_diff_idx": [int(idx[0]), int(idx[1])],
             "kernel_val_at_diff": float(kf_n[idx]) if diff.size else None,
             "ref_val_at_diff": float(LUt[idx]) if diff.size else None,
             "kernel_last_diag": float(kf_n[-1, -1]), "ref_last_diag": float(LUt[-1, -1]),
             "inv_allclose": bool(ok.all()), "inv_bad": int((~ok).sum()),
             "inv_nonfinite": int((~np.isfinite(kraw)).sum()),
             "inv_maxrel": float(np.nanmax(rel)) if np.isfinite(rel).any() else None,
             "kernel_inv": kraw.tolist(), "ref_inv": rin.tolist()}
        rec["batches"].append(e)
        lines.append(f"  b{b}: factor_exact={exact} ndiff={bad} firstdiff={e['first_diff_idx']} "
                     f"kernel={e['kernel_val_at_diff']} ref={e['ref_val_at_diff']} "
                     f"kdiag={e['kernel_last_diag']:.6e} rdiag={e['ref_last_diag']:.6e} "
                     f"inv_ok={e['inv_allclose']} bad={e['inv_bad']} nonfinite={e['inv_nonfinite']} "
                     f"maxrel={e['inv_maxrel']}")
    out["cases"][name] = rec

with open(os.path.join(ART, "probe_kernel_factors.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print("\n".join(lines))
