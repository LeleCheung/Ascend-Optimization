"""Diagnostic: identify the arithmetic of the reference torch.linalg.inv_ex on the target.

Compares the target reference (NPU) with CPU torch and with several float32
LAPACK-style LU emulations (numpy, no FMA contraction) to find which operation
sequence reproduces the reference bitwise / within contract tolerance.
"""

import json
import os

import numpy as np
import torch

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")

CASES = [
    ("c0", 1729, [3, 3], False),
    ("c1", 1730, [3, 3], True),
    ("c2", 1731, [2, 7, 7], False),
    ("c3", 1732, [2, 7, 7], True),
]


def gen(seed, shape, singular):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g)
    n = shape[-1]
    x = x + torch.eye(n) * (n + 1)
    if singular:
        x[..., -1, :] = x[..., 0, :]
    return x.to(torch.float32)


# ---------------------------------------------------------------- factorization
def lu_factor(Ain, mode="recip"):
    """netlib dgetf2: X(j,j) == 0 guard, pivot = first argmax |col|."""
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
                else:  # dscal with ONE/A(j,j)
                    A[j + 1:, j] = A[j + 1:, j] * (np.float32(1.0) / p)
                if mode == "fma":
                    prod = np.outer(
                        A[j + 1:, j].astype(np.float64), A[j, j + 1:].astype(np.float64)
                    )
                    upd = A[j + 1:, j + 1:].astype(np.float64) - prod
                    A[j + 1:, j + 1:] = upd.astype(np.float32)
                else:
                    A[j + 1:, j + 1:] = A[j + 1:, j + 1:] - np.outer(
                        A[j + 1:, j], A[j, j + 1:]
                    )
        elif info == 0:
            info = j + 1
    return A, piv, info


# ---------------------------------------------------------------------- solves
def solve_sub(LU, piv, div_mode="div", acc64=False):
    n = LU.shape[-1]
    X = np.eye(n, dtype=np.float32)

    def op(a, b, c, sub=True):
        # a <- a (+) b*c  elementwise outer product
        if acc64:
            r = a.astype(np.float64) - np.outer(
                b.astype(np.float64), c.astype(np.float64)
            )
            return r.astype(np.float32)
        return a - np.outer(b, c)

    for j in range(n):
        ip = int(piv[j])
        if ip != j:
            X[[j, ip], :] = X[[ip, j], :]
    for j in range(n - 1):
        X[j + 1:, :] = op(X[j + 1:, :], LU[j + 1:, j], X[j, :])
    for j in range(n - 1, -1, -1):
        if div_mode == "div":
            X[j, :] = X[j, :] / LU[j, j]
        else:
            X[j, :] = X[j, :] * (np.float32(1.0) / LU[j, j])
        if j > 0:
            X[:j, :] = op(X[:j, :], LU[:j, j], X[j, :])
    return X


def trtri_upper(M):
    n = M.shape[-1]
    A = np.array(M, dtype=np.float32, copy=True)
    for j in range(n - 1, -1, -1):
        if A[j, j] == 0:
            break
        A[j, j] = np.float32(1.0) / A[j, j]
        if j > 0:
            A[:j, j] = A[:j, j] * A[j, j]
        for k in range(j + 1, n):
            if A[j, k] != 0:
                t = A[j, k]
                A[:j, k] = A[:j, k] - t * A[:j, j]
    return A


def getri(LU, piv):
    """netlib dgetri: inv(U) by dtrtri, then column-by-column DGEMV solves."""
    n = LU.shape[-1]
    A = np.array(LU, dtype=np.float32, copy=True)
    U = np.triu(A)
    iU = trtri_upper(U)
    # L unit lower
    L = np.tril(A, -1) + np.eye(n, dtype=np.float32)
    # inv(L) column by column:  L x = e_k, forward substitution
    iL = np.zeros_like(L)
    for k in range(n):
        e = np.zeros(n, dtype=np.float32)
        e[k] = 1.0
        for j in range(n - 1):
            e[j + 1:] = e[j + 1:] - L[j + 1:, j] * e[j]
        iL[:, k] = e
    X = iU.copy()
    for j in range(n - 1, -1, -1):
        work = iL[j + 1:, j].copy()  # column of inv(L) below diagonal
        if j < n - 1:
            acc = X[:, j].astype(np.float32).copy()
            for k in range(j + 1, n):
                acc = acc - X[:, k] * work[k - j - 1]
            X[:, j] = acc
    return X


# --------------------------------------------------------------------- compare
def cmp(cand, ref, rtol=1e-4, atol=1e-4):
    c = np.asarray(cand, dtype=np.float32)
    r = np.asarray(ref, dtype=np.float32)
    if c.shape != r.shape:
        return {"shape": "mismatch"}
    d = np.abs(c.astype(np.float64) - r.astype(np.float64))
    tol = atol + rtol * np.abs(r.astype(np.float64))
    nbad = int((d > tol).sum())
    denom = np.abs(r.astype(np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(denom > 0, d / denom, np.where(d > 0, np.inf, 0.0))
    return {
        "n": int(c.size),
        "n_bad": nbad,
        "exact": int((c.view(np.int32) == r.view(np.int32)).sum()),
        "max_abs": float(d.max()),
        "max_rel": float(rel.max()),
        "ok": nbad == 0,
    }


def main():
    out = {"cases": []}
    dev = "cpu"
    try:
        import torch_npu  # noqa

        if hasattr(torch, "npu") and torch.npu.is_available():
            dev = "npu"
    except Exception as e:  # noqa
        out["dev_err"] = repr(e)
    out["torch"] = torch.__version__
    out["dev"] = dev
    try:
        import scipy

        out["scipy"] = scipy.__version__
    except Exception:
        out["scipy"] = None

    def refs(seed, shape, singular):
        A = gen(seed, shape, singular)
        r_cpu = torch.linalg.inv_ex(A)
        r_dev = torch.linalg.inv_ex(A.to(dev)) if dev != "cpu" else r_cpu
        return (
            A.numpy(),
            r_cpu.inverse.numpy(),
            int(r_cpu.info.sum().item()) if r_cpu.info.numel() > 1 else int(r_cpu.info.item()),
            r_dev.inverse.cpu().numpy(),
            r_dev.info.cpu().numpy(),
        )

    for name, seed, shape, singular in CASES:
        A, inv_cpu, info_cpu, inv_dev, info_dev = refs(seed, shape, singular)
        rec = {
            "case": name,
            "shape": shape,
            "singular": singular,
            "info_cpu": info_cpu,
            "info_dev": np.asarray(info_dev).tolist(),
            "cpu_eq_dev_exact": bool(
                (inv_cpu.view(np.int32) == inv_dev.view(np.int32)).all()
            ),
            "cpu_vs_dev": cmp(inv_cpu, inv_dev),
            "input": A.tolist(),
            "ref": inv_dev.tolist(),
            "variants": {},
        }
        # LAPACK factors for each LU mode
        mats = A.reshape(-1, A.shape[-1], A.shape[-1])
        for lumode in ("recip", "div", "fma"):
            for smode in ("div", "recip", "acc64"):
                for impl in ("sub", "getri"):
                    tag = f"lu-{lumode}_solve-{smode}_{impl}"
                    cand = np.zeros_like(mats)
                    infos = []
                    for b in range(mats.shape[0]):
                        try:
                            LU, piv, info = lu_factor(mats[b], lumode)
                            infos.append(info)
                            if impl == "getri":
                                cand[b] = getri(LU, piv)
                            else:
                                cand[b] = solve_sub(LU, piv, smode, acc64=(smode == "acc64"))
                        except Exception as e:  # noqa
                            cand[b] = np.nan
                            infos.append(-999)
                    rec["variants"][tag] = {
                        "cmp": cmp(cand.reshape(inv_dev.shape), inv_dev),
                        "info": infos,
                        "ref_info": np.asarray(info_dev).reshape(-1).tolist(),
                        "sample": cand.reshape(inv_dev.shape).tolist()
                        if name in ("c1", "c3")
                        else None,
                    }
            # U diagonal for the first batch
            LU, piv, info = lu_factor(mats[0], lumode)
            rec["udiag_" + lumode] = np.diag(LU).tolist()
            rec["piv_" + lumode] = piv.tolist()
        if out["scipy"]:
            try:
                import scipy.linalg as sla

                s_lu, s_piv = sla.lu_factor(mats[0].astype(np.float32))
                rec["scipy_udiag"] = np.diag(s_lu).tolist()
                rec["scipy_piv_raw"] = np.asarray(s_piv).tolist()
                LU_r, piv_r, _ = lu_factor(mats[0], "recip")
                LU_d, piv_d, _ = lu_factor(mats[0], "div")
                rec["lu_recip_eq_scipy"] = bool(
                    (LU_r.view(np.int32) == s_lu.view(np.int32)).all()
                )
                rec["lu_div_eq_scipy"] = bool(
                    (LU_d.view(np.int32) == s_lu.view(np.int32)).all()
                )
                s_inv = sla.inv(mats[0].astype(np.float32))
                rec["scipy_inv_vs_ref"] = cmp(s_inv, inv_dev.reshape(-1, *inv_dev.shape[-2:])[0])
            except Exception as e:  # noqa
                rec["scipy_err"] = repr(e)

        # compact stdout
        print(f"=== {name} {shape} singular={singular} info_cpu={info_cpu} "
              f"info_dev={np.asarray(info_dev).tolist()} cpu_eq_dev={rec['cpu_eq_dev_exact']}")
        print("   udiag recip:", rec.get("udiag_recip"))
        print("   udiag div  :", rec.get("udiag_div"))
        if out["scipy"]:
            print("   udiag scipy:", rec.get("scipy_udiag"), "pivraw", rec.get("scipy_piv_raw"))
            print("   eq scipy: recip", rec.get("lu_recip_eq_scipy"),
                  "div", rec.get("lu_div_eq_scipy"))
        rows = sorted(
            rec["variants"].items(),
            key=lambda kv: (kv[1]["cmp"].get("n_bad", 1e9), kv[1]["cmp"].get("max_rel", 1e9)),
        )
        for k, v in rows[:6]:
            print(f"   {k:34s} n_bad={v['cmp'].get('n_bad')} exact={v['cmp'].get('exact')}"
                  f" max_abs={v['cmp'].get('max_abs'):.6g} max_rel={v['cmp'].get('max_rel'):.3g}"
                  f" info={v['info']}")
        out["cases"].append(rec)

    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "probe_ref.json"), "w") as f:
        json.dump(out, f)
    print("ARTIFACT written:", os.path.join(ART, "probe_ref.json"))


main()
