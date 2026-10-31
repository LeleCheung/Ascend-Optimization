"""Diagnostic 4: search the float32 algorithm family that reproduces the CPU
reference inverse for exactly singular (duplicate-row) matrices.

The reference (torch.linalg.inv_ex == torch.linalg.inv == torch.linalg.solve)
returns finite huge values with info=0, so its elimination leaves a tiny
sub-ulp residual instead of an exact zero.  Candidate mechanisms:
  * LU + triangular solve (LAPACK sgetf2 style)
  * Gauss-Jordan with row normalization (pivot row divided by the pivot)
each with/without FMA contraction, reciprocal vs division, and both
orientations.  Score by max relative error against the reference.
"""

import json
import os

import numpy as np
import torch

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
F32 = np.float32


def gen(seed, shape, singular):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g)
    n = shape[-1]
    x = x + torch.eye(n) * (n + 1)
    if singular:
        x[..., -1, :] = x[..., 0, :]
    return x.to(torch.float32)


def fma(a, b, c):
    """float32 fma emulation."""
    return np.float32(np.float64(a) * np.float64(b) + np.float64(c))


def upd(a, b, c, use_fma):
    """a - b*c elementwise outer, float32."""
    if use_fma:
        if np.isscalar(b):
            return fma(np.float32(-1.0), b, a) if False else (a - fma(b, c, F32(0.0)))
        return fma(np.float32(-1.0), b[:, None], np.broadcast_to(a, (b.shape[0], a.shape[1] if a.ndim > 1 else 1)))
    return None


def lu_solve(Ain, mult="recip", use_fma=False, sdiv="div", pivot_last=False):
    A = np.array(Ain, dtype=F32, copy=True)
    n = A.shape[-1]
    piv = np.zeros(n, np.int32)
    info = 0
    for j in range(n):
        col = np.abs(A[j:, j])
        ip = j + (int(len(col) - 1 - np.argmax(col[::-1])) if pivot_last else int(np.argmax(col)))
        piv[j] = ip
        if ip != j:
            A[[j, ip], :] = A[[ip, j], :]
        p = A[j, j]
        if p != 0:
            if j < n - 1:
                if mult == "div":
                    A[j + 1:, j] = A[j + 1:, j] / p
                else:
                    A[j + 1:, j] = A[j + 1:, j] * (F32(1.0) / p)
                if use_fma:
                    prod = np.outer(A[j + 1:, j].astype(np.float64), A[j, j + 1:].astype(np.float64))
                    A[j + 1:, j + 1:] = (A[j + 1:, j + 1:].astype(np.float64) - prod).astype(F32)
                else:
                    A[j + 1:, j + 1:] = A[j + 1:, j + 1:] - np.outer(A[j + 1:, j], A[j, j + 1:])
        elif info == 0:
            info = j + 1
    if info:
        return None
    X = np.eye(n, dtype=F32)
    for j in range(n):
        ip = int(piv[j])
        if ip != j:
            X[[j, ip], :] = X[[ip, j], :]
    for j in range(n - 1):
        if use_fma:
            X[j + 1:, :] = (X[j + 1:, :].astype(np.float64)
                            - np.outer(A[j + 1:, j].astype(np.float64), X[j, :].astype(np.float64))).astype(F32)
        else:
            X[j + 1:, :] = X[j + 1:, :] - np.outer(A[j + 1:, j], X[j, :])
    for j in range(n - 1, -1, -1):
        if sdiv == "div":
            X[j, :] = X[j, :] / A[j, j]
        else:
            X[j, :] = X[j, :] * (F32(1.0) / A[j, j])
        if j > 0:
            if use_fma:
                X[:j, :] = (X[:j, :].astype(np.float64)
                            - np.outer(A[:j, j].astype(np.float64), X[j, :].astype(np.float64))).astype(F32)
            else:
                X[:j, :] = X[:j, :] - np.outer(A[:j, j], X[j, :])
    return X


def gauss_jordan(Ain, mult="recip", use_fma=False, normalize=True, pivot_last=False):
    """In-place Gauss-Jordan on [A | I]; row ops applied to both parts."""
    A = np.array(Ain, dtype=F32, copy=True)
    n = A.shape[-1]
    X = np.eye(n, dtype=F32)
    for j in range(n):
        col = np.abs(A[j:, j])
        ip = j + (int(len(col) - 1 - np.argmax(col[::-1])) if pivot_last else int(np.argmax(col)))
        if ip != j:
            A[[j, ip], :] = A[[ip, j], :]
            X[[j, ip], :] = X[[ip, j], :]
        p = A[j, j]
        if p == 0:
            return None
        if normalize:
            inv = F32(1.0) / p if mult == "recip" else None
            if inv is not None:
                A[j, :] = A[j, :] * inv
            else:
                A[j, :] = A[j, :] / p
            # the augmented part too, same op
            if inv is not None:
                X[j, :] = X[j, :] * inv
            else:
                X[j, :] = X[j, :] / p
            # eliminate
            for i in range(n):
                if i == j:
                    continue
                m = A[i, j]
                if m == 0:
                    continue
                if use_fma:
                    A[i, :] = (A[i, :].astype(np.float64)
                               - m.astype(np.float64) * A[j, :].astype(np.float64)).astype(F32)
                    X[i, :] = (X[i, :].astype(np.float64)
                               - m.astype(np.float64) * X[j, :].astype(np.float64)).astype(F32)
                else:
                    A[i, :] = A[i, :] - m * A[j, :]
                    X[i, :] = X[i, :] - m * X[j, :]
        else:
            for i in range(n):
                if i == j:
                    continue
                m = A[i, j] / p if mult == "div" else A[i, j] * (F32(1.0) / p)
                if use_fma:
                    A[i, :] = (A[i, :].astype(np.float64)
                               - m.astype(np.float64) * A[j, :].astype(np.float64)).astype(F32)
                    X[i, :] = (X[i, :].astype(np.float64)
                               - m.astype(np.float64) * X[j, :].astype(np.float64)).astype(F32)
                else:
                    A[i, :] = A[i, :] - m * A[j, :]
                    X[i, :] = X[i, :] - m * X[j, :]
            # no normalization: divide X row by the diagonal at the end
    if not normalize:
        for j in range(n):
            X[j, :] = X[j, :] / A[j, j]
    return X


def rel(cand, ref):
    c = np.asarray(cand, F32)
    r = np.asarray(ref, F32)
    d = np.abs(c.astype(np.float64) - r.astype(np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = np.where(np.abs(r) > 0, d / np.abs(r), np.where(d > 0, np.inf, 0.0))
    return float(np.nanmax(rr)), int((d > 1e-4 + 1e-4 * np.abs(r.astype(np.float64))).sum()), \
        int((c.view(np.int32) == r.view(np.int32)).sum()), int(r.size)


def main():
    variants = []
    for scheme in ("lu", "gj", "gjnn"):
        for mult in ("recip", "div"):
            for use_fma in (False, True):
                for sdiv in ("div", "recip"):
                    for tr in (False, True):
                        for plast in (False, True):
                            if scheme != "lu" and sdiv == "recip":
                                continue
                            variants.append((scheme, mult, use_fma, sdiv, tr, plast))

    cases = [("c0", 1729, [3, 3], False), ("c1", 1730, [3, 3], True),
             ("c2", 1731, [2, 7, 7], False), ("c3", 1732, [2, 7, 7], True)]
    data = {}
    for name, seed, shape, singular in cases:
        A = gen(seed, shape, singular)
        ref = torch.linalg.inv_ex(A)
        data[name] = (A.numpy().reshape(-1, shape[-1], shape[-1]),
                      ref.inverse.numpy().reshape(-1, shape[-1], shape[-1]),
                      ref.info.numpy().reshape(-1).tolist())
        print(name, "info", data[name][2])

    results = []
    for v in variants:
        scheme, mult, use_fma, sdiv, tr, plast = v
        tag = f"{scheme}_m{mult}_fma{int(use_fma)}_sd{sdiv}_T{int(tr)}_pl{int(plast)}"
        worst = {}
        ok = True
        for name in ("c0", "c1", "c2", "c3"):
            mats, refs, _ = data[name]
            out = np.zeros_like(mats)
            good = True
            for b in range(mats.shape[0]):
                M = mats[b]
                B = M.T.copy() if tr else M
                if scheme == "lu":
                    X = lu_solve(B, mult, use_fma, sdiv, plast)
                else:
                    X = gauss_jordan(B, mult, use_fma, normalize=(scheme == "gj"), pivot_last=plast)
                if X is None:
                    good = False
                    break
                out[b] = X.T if tr else X
            if not good:
                ok = False
                break
            worst[name] = rel(out, refs)
        if not ok:
            continue
        results.append((max(worst[k][0] for k in worst), tag, worst))
    results.sort(key=lambda x: x[0])
    print("\nTop variants (by worst-case max_rel over all 4 cases):")
    for w, tag, per in results[:12]:
        print(f"  {tag:34s} worst={w:.4g}  " + "  ".join(
            f"{k}:rel={per[k][0]:.3g}/bad={per[k][1]}/ex={per[k][2]}/{per[k][3]}" for k in per))

    # detail of the best
    if results:
        w, tag, per = results[0]
        parts = tag.split("_")
        scheme = parts[0]
        mult = parts[1][1:]
        use_fma = parts[2][3:] == "1"
        sdiv = parts[3][2:]
        tr = parts[4][1:] == "1"
        plast = parts[5][2:] == "1"
        detail = {}
        for name in ("c1", "c3"):
            mats, refs, _ = data[name]
            out = np.zeros_like(mats)
            for b in range(mats.shape[0]):
                M = mats[b]
                B = M.T.copy() if tr else M
                X = (lu_solve(B, mult, use_fma, sdiv, plast) if scheme == "lu"
                     else gauss_jordan(B, mult, use_fma, scheme == "gj", plast))
                out[b] = X.T if tr else X
            detail[name] = {"cand": out.tolist(), "ref": refs.tolist()}
            print(f"  {name} cand[0][0]={out[0, 0].tolist()}")
            print(f"  {name} ref [0][0]={refs[0, 0].tolist()}")
        os.makedirs(ART, exist_ok=True)
        with open(os.path.join(ART, "best_variant.json"), "w") as f:
            json.dump({"tag": tag, "detail": detail}, f)
        print("ARTIFACT best_variant.json", tag)


main()
