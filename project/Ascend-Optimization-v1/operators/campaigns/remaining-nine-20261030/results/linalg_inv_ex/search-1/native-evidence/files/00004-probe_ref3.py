"""Diagnostic 3: identify the code path behind the CPU reference of linalg_inv_ex.

For singular inputs torch.linalg.inv_ex returns finite huge values with info=0
while lu_factor_ex reports an exact zero pivot (info=n).  This probes which aten
path produces the inverse and whether the values are an LU solve with a tiny
nonzero last pivot.
"""

import json
import os

import numpy as np
import torch

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
torch.set_printoptions(precision=17)

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
                A[j + 1:, j + 1:] = A[j + 1:, j + 1:] - np.outer(
                    A[j + 1:, j], A[j, j + 1:]
                )
        elif info == 0:
            info = j + 1
    return A, piv, info


def solve_sub(LU, piv, sdiv="div"):
    n = LU.shape[-1]
    X = np.eye(n, dtype=np.float32)
    for j in range(n):
        ip = int(piv[j])
        if ip != j:
            X[[j, ip], :] = X[[ip, j], :]
    for j in range(n - 1):
        X[j + 1:, :] = X[j + 1:, :] - np.outer(LU[j + 1:, j], X[j, :])
    for j in range(n - 1, -1, -1):
        if sdiv == "div":
            X[j, :] = X[j, :] / LU[j, j]
        else:
            X[j, :] = X[j, :] * (np.float32(1.0) / LU[j, j])
        if j > 0:
            X[:j, :] = X[:j, :] - np.outer(LU[:j, j], X[j, :])
    return X


def relerr(cand, ref):
    c = np.asarray(cand, dtype=np.float32).astype(np.float64)
    r = np.asarray(ref, dtype=np.float32).astype(np.float64)
    if c.shape != r.shape:
        return None
    d = np.abs(c - r)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(np.abs(r) > 0, d / np.abs(r), np.where(d > 0, np.inf, 0.0))
    return {
        "max_rel": float(np.nanmax(rel)),
        "max_abs": float(np.nanmax(d)),
        "nbad_1e4": int((d > 1e-4 + 1e-4 * np.abs(r)).sum()),
        "exact": int(
            (np.asarray(cand, np.float32).view(np.int32) == np.asarray(ref, np.float32).view(np.int32)).sum()
        ),
        "n": int(np.asarray(ref).size),
    }


def main():
    out = {}
    print("torch", torch.__version__)
    try:
        print("config:", torch.__config__.show().replace("\n", " | ")[:600])
    except Exception as e:  # noqa
        print("cfg err", e)

    # --- info semantics on obviously singular matrices
    probes = {
        "zeros3": np.zeros((3, 3), np.float32),
        "rank1": np.ones((3, 3), np.float32),
        "dup_rows": np.array([[2.0, 1.0, 0.5], [0.3, 3.0, 1.0], [2.0, 1.0, 0.5]], np.float32),
        "diag0": np.diag(np.array([1.0, 0.0, 2.0], np.float32)),
    }
    for k, v in probes.items():
        t = torch.from_numpy(v.copy())
        r = torch.linalg.inv_ex(t)
        lu = torch.linalg.lu_factor_ex(t)
        out["info_probe_" + k] = {
            "inv": r.inverse.tolist(),
            "info": r.info.tolist(),
            "lu_info": lu[2].tolist(),
            "lu_diag": torch.diagonal(lu[0]).tolist(),
        }
        print(f"[{k}] inv_ex info={r.info.tolist()} lu info={lu[2].tolist()} "
              f"udiag={torch.diagonal(lu[0]).tolist()}")
        print("    inv =", r.inverse.reshape(-1).tolist()[:6])

    for name, seed, shape, singular in CASES:
        A = gen(seed, shape, singular)
        n = shape[-1]
        I = torch.eye(n).expand(shape).contiguous()
        ref = torch.linalg.inv_ex(A)
        refx = ref.inverse
        mats = A.numpy().reshape(-1, n, n)
        refn = refx.numpy().reshape(-1, n, n)

        rec = {
            "input": A.tolist(),
            "ref": refx.tolist(),
            "info": ref.info.tolist(),
            "candidates": {},
        }
        cands = {}
        try:
            cands["inv"] = torch.linalg.inv(A)
        except Exception as e:  # noqa
            rec["inv_err"] = repr(e)
        try:
            cands["solve_I"] = torch.linalg.solve(A, I)
        except Exception as e:  # noqa
            rec["solve_err"] = repr(e)
        try:
            cands["solve_ex_I"] = torch.linalg.solve_ex(A, I).result
        except Exception as e:  # noqa
            rec["solve_ex_err"] = repr(e)
        try:
            LU, piv, inf = torch.linalg.lu_factor_ex(A)
            rec["lu_info"] = inf.tolist()
            cands["lu_solve_ex"] = torch.linalg.lu_solve(LU, piv, I)
            LU2, piv2 = torch.linalg.lu_factor(A)
            cands["lu_solve"] = torch.linalg.lu_solve(LU2, piv2, I)
        except Exception as e:  # noqa
            rec["lu_solve_err"] = repr(e)
        try:
            cands["pinv"] = torch.linalg.pinv(A)
        except Exception as e:  # noqa
            rec["pinv_err"] = repr(e)
        try:
            cands["lstsq"] = torch.linalg.lstsq(A, I).solution
        except Exception as e:  # noqa
            rec["lstsq_err"] = repr(e)
        try:
            cands["solve_At_T"] = torch.linalg.solve(A.transpose(-1, -2), I).transpose(-1, -2)
        except Exception as e:  # noqa
            rec["solve_At_err"] = repr(e)

        print(f"=== {name} {shape} sing={singular} inv_ex.info={rec['info']} lu.info={rec.get('lu_info')}")
        for k, v in cands.items():
            r = relerr(v.numpy(), refx.numpy())
            rec["candidates"][k] = r
            if r:
                print(f"    {k:14s} bitwise_eq={r['exact'] == r['n']} max_rel={r['max_rel']:.4g} "
                      f"nbad={r['nbad_1e4']} vals={v.reshape(-1)[:3].tolist()}")

        # --- model check: LU (bitwise equal to torch's) + tiny final pivot fitted from ref
        for b in range(mats.shape[0]):
            M = mats[b]
            LU, piv, info = lu_factor(M, "recip")
            diag = np.diag(LU).astype(np.float64)
            zero_idx = [int(i) for i in range(n) if LU[i, i] == 0.0]
            rec[f"b{b}_lu_diag"] = diag.tolist()
            rec[f"b{b}_lu_zero_idx"] = zero_idx
            if len(zero_idx) != 1:
                print(f"    b{b} zero diag count {len(zero_idx)} -> skip model")
                continue
            zi = zero_idx[0]
            Xr = refn[b]
            # model: X[n-1,j] = b3(j)/p ; for zi == n-1 b3 = row of Linv
            p_from_diag = np.float32(1.0 / np.float64(Xr[zi, zi]))
            LUp = LU.copy()
            LUp[zi, zi] = p_from_diag
            Xm = solve_sub(LUp, piv, "div")
            r = relerr(Xm, Xr)
            rec[f"b{b}_model_p"] = float(p_from_diag)
            rec[f"b{b}_model"] = r
            rec[f"b{b}_model_mat"] = Xm.tolist()
            print(f"    b{b} zero_diag_at={zi} p={p_from_diag:.10e} model: {r}")
            print(f"       ref  last row: {Xr[zi].tolist()}")
            print(f"       model last row: {Xm[zi].tolist()}")
        out[name] = rec

    os.makedirs(ART, exist_ok=True)
    p = os.path.join(ART, "probe_ref3.json")
    with open(p, "w") as f:
        json.dump(out, f)
    print("ARTIFACT", p)


main()
