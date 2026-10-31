"""Diagnostic 2: pin down the exact arithmetic of CPU torch.linalg.inv_ex.

Hypothesis: ATen calls column-major LAPACK on row-major data, i.e. it
effectively factorizes A^T.  A^T has no exactly-duplicated rows, so the last
pivot is a tiny nonzero float (info=0, inverse ~1e8) instead of an exact 0.

Tests: dispatch trace, torch lu_factor_ex factors (A and A^T) vs numpy
emulations, and many inv emulation variants vs the reference inverse.
"""

import json
import os

import numpy as np
import torch

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
CASES = [
    ("c1", 1730, [3, 3], True),
    ("c3", 1732, [2, 7, 7], True),
    ("c0", 1729, [3, 3], False),
]


def gen(seed, shape, singular):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g)
    n = shape[-1]
    x = x + torch.eye(n) * (n + 1)
    if singular:
        x[..., -1, :] = x[..., 0, :]
    return x.to(torch.float32)


def lu_factor(Ain, mode="recip"):
    A = np.array(Ain, dtype=np.float32, copy=True)
    n = A.shape[-1]
    piv = np.zeros(n, dtype=np.int32)
    info = 0
    for j in range(n):
        ip = j + int(np.argmax(np.abs(A[j:, j])))
        piv[j] = ip
        if ip != j:
            A[[j, ip], :] = A[[ip, j], :]
        p = A[j, j]
        if p != 0:
            if j < n - 1:
                if mode == "div":
                    A[j + 1:, j] = A[j + 1:, j] / p
                else:
                    A[j + 1:, j] = A[j + 1:, j] * (np.float32(1.0) / p)
                if mode == "fma":
                    prod = np.outer(
                        A[j + 1:, j].astype(np.float64), A[j, j + 1:].astype(np.float64)
                    )
                    A[j + 1:, j + 1:] = (
                        A[j + 1:, j + 1:].astype(np.float64) - prod
                    ).astype(np.float32)
                else:
                    A[j + 1:, j + 1:] = A[j + 1:, j + 1:] - np.outer(
                        A[j + 1:, j], A[j, j + 1:]
                    )
        elif info == 0:
            info = j + 1
    return A, piv, info


def solve_sub(LU, piv, div_mode="div", fma=False):
    n = LU.shape[-1]
    X = np.eye(n, dtype=np.float32)

    def upd(a, b, c):
        if fma:
            return (a.astype(np.float64) - np.outer(b.astype(np.float64), c.astype(np.float64))).astype(np.float32)
        return a - np.outer(b, c)

    for j in range(n):
        ip = int(piv[j])
        if ip != j:
            X[[j, ip], :] = X[[ip, j], :]
    for j in range(n - 1):
        X[j + 1:, :] = upd(X[j + 1:, :], LU[j + 1:, j], X[j, :])
    for j in range(n - 1, -1, -1):
        if div_mode == "div":
            X[j, :] = X[j, :] / LU[j, j]
        else:
            X[j, :] = X[j, :] * (np.float32(1.0) / LU[j, j])
        if j > 0:
            X[:j, :] = upd(X[:j, :], LU[:j, j], X[j, :])
    return X


def emu_inv(M, lu_mode, sdiv, fma_solve, transpose):
    B = M.T if transpose else M
    LU, piv, info = lu_factor(B, lu_mode)
    if info != 0:
        return None, info, LU, piv
    XB = solve_sub(LU, piv, sdiv, fma_solve)
    return (XB.T if transpose else XB), info, LU, piv


def cmp(cand, ref, rtol=1e-4, atol=1e-4):
    c = np.asarray(cand, dtype=np.float32)
    r = np.asarray(ref, dtype=np.float32)
    d = np.abs(c.astype(np.float64) - r.astype(np.float64))
    tol = atol + rtol * np.abs(r.astype(np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(np.abs(r) > 0, d / np.abs(r.astype(np.float64)), np.where(d > 0, np.inf, 0))
    return {
        "n_bad": int((d > tol).sum()),
        "n": int(c.size),
        "exact": int((c.view(np.int32) == r.view(np.int32)).sum()),
        "max_rel": float(np.nanmax(rel)),
        "max_abs": float(np.nanmax(d)),
    }


def trace_ops(A):
    names = []
    try:
        from torch.utils._python_dispatch import TorchDispatchMode

        class Rec(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                names.append(str(func))
                return func(*args, **(kwargs or {}))

        with Rec():
            torch.linalg.inv_ex(A)
    except Exception as e:  # noqa
        names.append("TRACE_ERR " + repr(e))
    return names


def main():
    out = {"cases": []}
    print("torch", torch.__version__)
    A0 = gen(1730, [3, 3], True)
    print("trace inv_ex:", trace_ops(A0))

    for name, seed, shape, singular in CASES:
        A = gen(seed, shape, singular)
        ref = torch.linalg.inv_ex(A).inverse.numpy()
        mats = A.numpy().reshape(-1, shape[-1], shape[-1])
        rec = {"case": name, "shape": shape, "ref": ref.tolist(), "variants": {}}

        # --- torch's own LU factors, per batch element, of A and of A^T
        try:
            lu_t, piv_t, info_t = torch.linalg.lu_factor_ex(A)
            rec["lu_torch_0"] = lu_t.reshape(-1, shape[-1], shape[-1])[0].numpy().tolist()
            rec["piv_torch_0"] = piv_t.reshape(-1, shape[-1])[0].numpy().tolist()
            rec["info_torch"] = info_t.numpy().reshape(-1).tolist()
            print(f"=== {name} {shape} sing={singular} torch lu info={rec['info_torch']}"
                  f" piv0={rec['piv_torch_0']}")
            print("   torch LU row0:", np.round(np.array(rec["lu_torch_0"])[0], 8).tolist())
            print("   torch U diag :", np.diag(np.array(rec["lu_torch_0"])).tolist())
        except Exception as e:  # noqa
            rec["lu_torch_err"] = repr(e)
            print("   lu_factor_ex err", repr(e))

        for b in range(mats.shape[0]):
            M = mats[b]
            LUa, pa, ia = lu_factor(M, "recip")
            LUt, pt, it = lu_factor(M.T.copy(), "recip")
            rec[f"b{b}_emuA_udiag"] = np.diag(LUa).tolist()
            rec[f"b{b}_emuT_udiag"] = np.diag(LUt).tolist()
            if b == 0:
                print("   emuA(recip) Udiag:", np.diag(LUa).tolist(), "info", ia, "piv", pa.tolist())
                print("   emuT(recip) Udiag:", np.diag(LUt).tolist(), "info", it, "piv", pt.tolist())
            try:
                t = torch.linalg.lu_factor_ex(torch.from_numpy(M.copy()))
                tlu = t[0].numpy()
                rec[f"b{b}_torch_eq_emuA"] = bool((tlu.view(np.int32) == LUa.view(np.int32)).all())
                rec[f"b{b}_torch_eq_emuT"] = bool((tlu.view(np.int32) == LUt.view(np.int32)).all())
                if b == 0:
                    print("   torch LU == emuA:", rec[f"b{b}_torch_eq_emuA"],
                          " == emuT:", rec[f"b{b}_torch_eq_emuT"])
            except Exception as e:  # noqa
                rec[f"b{b}_torch_lu_err"] = repr(e)

        best = []
        for tr in (False, True):
            for lumode in ("recip", "div", "fma"):
                for sdiv in ("div", "recip"):
                    for fma_solve in (False, True):
                        cand = np.zeros_like(mats)
                        ok = True
                        for b in range(mats.shape[0]):
                            X, info, _, _ = emu_inv(mats[b], lumode, sdiv, fma_solve, tr)
                            if X is None:
                                ok = False
                                break
                            cand[b] = X
                        tag = f"T{int(tr)}_lu{lumode}_sdiv{sdiv}_fma{int(fma_solve)}"
                        if not ok:
                            rec["variants"][tag] = {"failed": "exact zero pivot"}
                            continue
                        c = cmp(cand.reshape(ref.shape), ref)
                        rec["variants"][tag] = c
                        best.append((c["n_bad"], c["max_rel"], tag, c["exact"]))
        best.sort(key=lambda x: (x[0], x[1]))
        print("   ref[0:2,0]:", np.array(rec["ref"]).reshape(-1, *ref.shape[-2:])[0][:2, 0].tolist())
        for nb, mr, tag, ex in best[:8]:
            print(f"   {tag:28s} n_bad={nb:3d} max_rel={mr:.3g} exact={ex}")
        # print best candidate's first matrix for the singular case
        if best:
            tagb = best[0][2]
            tr = tagb[1] == "1"
            parts = tagb.split("_")
            lumode = parts[1][2:]
            sdiv = parts[2][4:]
            fma_solve = parts[3][3:] == "1"
            X, _, _, _ = emu_inv(mats[0], lumode, sdiv, fma_solve, tr)
            rec["best_tag"] = tagb
            rec["best_mat0"] = None if X is None else X.tolist()
            if X is not None:
                print("   best mat0 row0:", X[0].tolist())
        out["cases"].append(rec)

    os.makedirs(ART, exist_ok=True)
    p = os.path.join(ART, "probe_ref2.json")
    with open(p, "w") as f:
        json.dump(out, f)
    print("ARTIFACT", p)


main()
