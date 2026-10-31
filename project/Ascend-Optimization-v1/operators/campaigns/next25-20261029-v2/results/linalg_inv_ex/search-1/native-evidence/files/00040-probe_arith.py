"""Diagnostic: recover the exact fp32 arithmetic of the target reference LU/solve.

The reference op falls back to CPU LAPACK. We read its LU factors through
torch.linalg.lu_factor_ex and test which scalar fp32 operation ordering
(multiplier by division vs reciprocal-multiply; trailing update by mul+sub vs FMA;
inverse by LU-solve vs getri-style) reproduces both the factors and the inverse.
"""
import json
import os

import numpy as np
import torch

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")


def gen_inputs(spec, seed):
    dtype = getattr(torch, spec["dtype"])
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(spec["shape"], generator=gen)
    x = x + torch.eye(x.shape[-1], dtype=dtype) * (x.shape[-1] + 1)
    if spec.get("singular"):
        x[..., -1, :] = x[..., 0, :]
    return x.to(dtype=dtype)


def f32(v):
    return np.float32(v)


def lu_emulate(a, mult, upd):
    work = np.array(a, dtype=np.float32, copy=True)
    n = work.shape[0]
    piv = []
    info = 0
    for j in range(n):
        p = int(np.argmax(np.abs(work[j:, j]))) + j
        piv.append(p)
        if p != j:
            work[[j, p], :] = work[[p, j], :]
        pv = work[j, j]
        if pv != 0.0:
            if mult == "div":
                l = (work[j + 1:, j] / pv).astype(np.float32)
            else:
                rcp = np.float32(1.0) / pv
                l = (work[j + 1:, j] * rcp).astype(np.float32)
            work[j + 1:, j] = l
            if j + 1 < n:
                r0 = work[j, j + 1:]
                sub = work[j + 1:, j + 1:]
                if upd == "mulsub":
                    work[j + 1:, j + 1:] = (sub - l[:, None] * r0[None, :]).astype(np.float32)
                else:  # fma: exact product in float64, single rounding
                    work[j + 1:, j + 1:] = (
                        sub.astype(np.float64) - l[:, None].astype(np.float64) * r0[None, :].astype(np.float64)
                    ).astype(np.float32)
        elif info == 0:
            info = j + 1
    return work, piv, info


def solve_emu(work, piv, rhs, mult, upd, style="solve"):
    n = work.shape[0]
    m = np.array(rhs, dtype=np.float32, copy=True)
    L = np.tril(work, -1) + np.eye(n, dtype=np.float32)
    U = np.triu(work)
    P = np.eye(n, dtype=np.float32)
    for j, p in enumerate(piv):
        if p != j:
            P[[j, p], :] = P[[p, j], :]
    b = P @ m if style != "solve" else m
    if style == "getri":
        # invert U then L, then invU @ invL
        Uinv = np.eye(n, dtype=np.float32)
        for j in range(n - 1, -1, -1):
            if U[j, j] == 0.0:
                Uinv[j, :] = np.float32(0)
                continue
            r = Uinv[j, :].copy()
            if j + 1 < n:
                r = (r - U[j, j + 1:] @ Uinv[j + 1:, :]).astype(np.float32) if False else r
                for k in range(j + 1, n):
                    r = (r - U[j, k] * Uinv[k, :]).astype(np.float32)
            Uinv[j, :] = (r / U[j, j]).astype(np.float32)
        Linv = np.eye(n, dtype=np.float32)
        for j in range(n):
            for i in range(j + 1, n):
                s = np.float32(0)
                for k in range(j, i):
                    s = np.float32(s + Linv[i, k] * L[k, j])
                Linv[i, j] = np.float32(-s)
        inv = (Uinv @ np.eye(n, dtype=np.float32))
        # inv = invU @ invL  (with permutation applied on the left)
        full = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                acc = np.float64(0.0)
                for k in range(n):
                    acc += np.float64(Uinv[i, k]) * np.float64(Linv[k, j])
                full[i, j] = np.float32(acc)
        inv = full
        out = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                acc = np.float64(0.0)
                for k in range(n):
                    acc += np.float64(P[i, k]) * np.float64(inv[k, j])
                out[i, j] = np.float32(acc)
        return out
    # forward substitution L y = P b   (unit lower)
    for j in range(n):
        if j + 1 < n:
            l = work[j + 1:, j]
            if upd == "mulsub":
                b[j + 1:, :] = (b[j + 1:, :] - l[:, None] * b[j][None, :]).astype(np.float32)
            else:
                b[j + 1:, :] = (
                    b[j + 1:, :].astype(np.float64) - l[:, None].astype(np.float64) * b[j][None, :].astype(np.float64)
                ).astype(np.float32)
    # back substitution U x = y
    for j in range(n - 1, -1, -1):
        d = work[j, j]
        b[j, :] = (b[j, :] / d).astype(np.float32)
        if j > 0:
            u = work[:j, j]
            if upd == "mulsub":
                b[:j, :] = (b[:j, :] - u[:, None] * b[j][None, :]).astype(np.float32)
            else:
                b[:j, :] = (
                    b[:j, :].astype(np.float64) - u[:, None].astype(np.float64) * b[j][None, :].astype(np.float64)
                ).astype(np.float32)
    return b


def cmp(ref, cand):
    ref = np.asarray(ref, dtype=np.float32)
    cand = np.asarray(cand, dtype=np.float32)
    ok = np.isclose(ref, cand, rtol=1e-4, atol=1e-4, equal_nan=True)
    denom = np.maximum(np.abs(ref), np.float32(1e-30))
    rel = np.abs(cand - ref) / denom
    return {
        "allclose_1e-4": bool(ok.all()),
        "matched_frac": float(ok.mean()),
        "max_rel": float(rel.max()),
        "n_bad": int((~ok).sum()),
    }


cases = [
    ("c1_sing3", {"dtype": "float32", "shape": [3, 3], "singular": True}, 1730),
    ("c3_sing7", {"dtype": "float32", "shape": [2, 7, 7], "singular": True}, 1732),
    ("c0_reg3", {"dtype": "float32", "shape": [3, 3], "singular": False}, 1729),
]

out = {"env": {"torch": torch.__version__, "mkl": bool(getattr(torch.backends, "mkl", None) and torch.backends.mkl.is_available())}, "cases": {}}

for name, spec, seed in cases:
    A = gen_inputs(spec, seed)
    Af = A.reshape(-1, spec["shape"][-1], spec["shape"][-1])
    ref = torch.linalg.inv_ex(A, check_errors=False)
    LU, pivots, lu_info = torch.linalg.lu_factor_ex(A)
    LUf = LU.reshape(-1, *LU.shape[-2:])
    rec = {
        "shape": spec["shape"],
        "inv_info_ref": ref.info.tolist(),
        "lu_factor_info": lu_info.tolist(),
        "lu_pivots": (pivots + 1).tolist(),
        "ref_inv": [[float(v) for v in row] for row in
                    ref.inverse.reshape(-1, ref.inverse.shape[-1]).tolist()] if ref.inverse.numel() <= 64 else None,
        "lu_diag": [float(LUf[b].numpy().diagonal()[-1]) for b in range(LUf.shape[0])],
        "batches": [],
    }
    for b in range(Af.shape[0]):
        a = Af[b].numpy()
        lu_t = LUf[b].numpy()
        entry = {"b": b, "lu_last_diag_torch": float(lu_t[-1, -1]),
                 "inv_ref": ref.inverse.reshape(-1, *spec["shape"][-2:])[b].tolist()}
        # which multiplier arithmetic: compare L[2,0] style entries against variants
        pv = a[0, 0]
        cands = {
            "div": float(np.float32(a[-1, 0] / pv)),
            "rcp": float(np.float32(a[-1, 0] * np.float32(np.float32(1.0) / pv))),
            "torch_L": float(lu_t[-1, 0]),
        }
        entry["mult_test_col0"] = cands
        variants = {}
        for mult in ("div", "rcp"):
            for upd in ("mulsub", "fma"):
                wk, piv, info = lu_emulate(a, mult, upd)
                entry.setdefault("lu_variants", {})[f"{mult}/{upd}"] = {
                    "info": info,
                    "last_diag": float(wk[-1, -1]),
                    "lu_max_abs_diff_vs_torch": float(np.abs(wk - lu_t).max()),
                    "lu_exact_match": bool(np.array_equal(wk, lu_t)),
                    "pivots": piv,
                }
                inv_s = solve_emu(wk, piv, np.eye(a.shape[0], dtype=np.float32), mult, upd, "solve")
                inv_g = solve_emu(wk, piv, np.eye(a.shape[0], dtype=np.float32), mult, upd, "getri")
                variants[f"solve/{mult}/{upd}"] = cmp(entry["inv_ref"], inv_s)
                variants[f"getri/{mult}/{upd}"] = cmp(entry["inv_ref"], inv_g)
                entry.setdefault("inv_variant_vals", {})[f"solve/{mult}/{upd}"] = inv_s.tolist()
        entry["inv_variant_cmp"] = variants
        # solve from torch's own factors (isolate the solve step)
        piv_t = (pivots[b].numpy() - 1).tolist()
        for upd in ("mulsub", "fma"):
            inv_s = solve_emu(lu_t.copy(), piv_t, np.eye(a.shape[0], dtype=np.float32), "div", upd, "solve")
            variants[f"torchfactors_solve/{upd}"] = cmp(entry["inv_ref"], inv_s)
            inv_g = solve_emu(lu_t.copy(), piv_t, np.eye(a.shape[0], dtype=np.float32), "div", upd, "getri")
            variants[f"torchfactors_getri/{upd}"] = cmp(entry["inv_ref"], inv_g)
            entry.setdefault("inv_variant_vals", {})[f"torchfactors_solve/{upd}"] = inv_s.tolist()
        rec["batches"].append(entry)
    out["cases"][name] = rec

with open(os.path.join(ART, "probe_arith.json"), "w") as fh:
    json.dump(out, fh, indent=1)

summary = {}
for name, rec in out["cases"].items():
    best = []
    for b in rec["batches"]:
        for k, v in b["inv_variant_cmp"].items():
            if v["allclose_1e-4"]:
                best.append(f"b{b}:{k}")
        summary.setdefault(name, {})[f"b{b}_mult_test"] = b["mult_test_col0"]
        summary[name][f"b{b}_lu_variants"] = b["lu_variants"]
        summary[name][f"b{b}_inv_cmp"] = {k: (v["allclose_1e-4"], round(v["max_rel"], 6), v["n_bad"])
                                          for k, v in b["inv_variant_cmp"].items()}
        summary[name][f"b{b}_lu_diag_torch"] = b["lu_last_diag_torch"]
    summary[name]["exact_matches"] = best
print(json.dumps(summary, indent=1))
