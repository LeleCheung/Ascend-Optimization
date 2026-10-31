"""Pin down the arithmetic of torch 2.9 CPU linalg_inv_ex on a singular input.

Facts already established:
  * reference == CPU fallback (bit-identical to npu_reference in the audit)
  * A rows 0 and -1 are bit-identical; LAPACK-style partial-pivoting LU on A
    gives U[n-1,n-1] == 0.0 exactly and lu_factor_ex info = n
  * but linalg_inv_ex returns info = 0 and finite ~1e8..1e9 entries

So inv_ex does not use that LU.  This script tests:
  (a) bit-exact agreement with other torch CPU callables
  (b) whether the reference is LU-based on the transpose (A^T has identical
      columns 0 and -1, which does not cancel exactly)
  (c) Gauss-Jordan with pivot-row normalisation (breaks exact cancellation)
  (d) if LU-based, which tiny pivot value u22 the reference implies
"""

import json
import os

import numpy as np
import torch

OUT = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")


def build(spec, seed):
    n = spec["shape"][-1]
    g = torch.Generator().manual_seed(seed)
    _ = torch.randn(spec["shape"], generator=g)
    x = torch.randn(spec["shape"], generator=g)
    x = x + torch.eye(n) * (n + 1)
    if spec.get("singular"):
        x[..., -1, :] = x[..., 0, :]
    return x.to(torch.float32).contiguous()


def upd_fma(m, y, cur):
    return np.float32(np.float64(cur) - np.float64(m) * np.float64(y))


def upd_mul(m, y, cur):
    return np.float32(cur - np.float32(m * y))


UPD = {"fma": upd_fma, "mul": upd_mul}


def emu_lu(A, upd, scale="recip"):
    n = A.shape[0]
    M = np.array(A, dtype=np.float32, copy=True)
    piv = np.zeros(n, np.int32)
    info = 0
    for j in range(n):
        jp = j + int(np.argmax(np.abs(M[j:, j])))
        piv[j] = jp
        if M[jp, j] != 0.0:
            if jp != j:
                M[[j, jp], :] = M[[jp, j], :]
            if j < n - 1:
                if scale == "recip":
                    rec = np.float32(1.0) / np.float32(M[j, j])
                    M[j + 1:, j] = np.float32(M[j + 1:, j] * rec)
                else:
                    pv = np.float32(M[j, j])
                    M[j + 1:, j] = np.float32(M[j + 1:, j] / pv)
        else:
            if info == 0:
                info = j + 1
        for k in range(j + 1, n):
            y = np.float32(M[j, k])
            for i in range(j + 1, n):
                M[i, k] = upd(M[i, j], y, M[i, k])
    return M, piv, info


def emu_solve(LU, piv, upd, back="div"):
    n = LU.shape[0]
    X = np.zeros((n, n), np.float32)
    for k in range(n):
        b = np.zeros(n, np.float32)
        b[k] = np.float32(1.0)
        for j in range(n):
            p = int(piv[j])
            if p != j:
                b[j], b[p] = b[p], b[j]
        for j in range(n):
            t = np.float32(b[j])
            if t != 0.0:
                for i in range(j + 1, n):
                    b[i] = upd(LU[i, j], t, b[i])
        for j in range(n - 1, -1, -1):
            if back == "div":
                t = np.float32(b[j] / LU[j, j])
            else:
                t = np.float32(b[j] * (np.float32(1.0) / np.float32(LU[j, j])))
            b[j] = t
            if t != 0.0:
                for i in range(j):
                    b[i] = upd(LU[i, j], t, b[i])
        X[:, k] = b
    return X


def emu_gj(A, upd, norm="div"):
    """Gauss-Jordan on [A|I] with pivot-row normalisation."""
    n = A.shape[0]
    M = np.array(A, dtype=np.float32, copy=True)
    X = np.eye(n, dtype=np.float32)
    for j in range(n):
        jp = j + int(np.argmax(np.abs(M[j:, j])))
        if jp != j:
            M[[j, jp], :] = M[[jp, j], :]
            X[[j, jp], :] = X[[jp, j], :]
        p = np.float32(M[j, j])
        if norm == "div":
            M[j, :] = np.float32(M[j, :] / p)
            X[j, :] = np.float32(X[j, :] / p)
        else:
            rec = np.float32(1.0) / p
            M[j, :] = np.float32(M[j, :] * rec)
            X[j, :] = np.float32(X[j, :] * rec)
        for i in range(n):
            if i == j:
                continue
            f = np.float32(M[i, j])
            if f != 0.0:
                for k in range(n):
                    M[i, k] = upd(f, M[j, k], M[i, k])
                for k in range(n):
                    X[i, k] = upd(f, X[j, k], X[i, k])
    return X


def bits_eq(a, b):
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    b = np.ascontiguousarray(np.asarray(b, dtype=np.float32))
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    exact = int(np.sum(a.view(np.int32) == b.view(np.int32)))
    d = np.abs(np.float64(a) - np.float64(b))
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(np.float64(np.abs(b)) > 0, d / np.abs(np.float64(b)), d)
    return {
        "exact": exact,
        "numel": int(a.size),
        "max_abs": float(np.nanmax(d)) if a.size else 0.0,
        "max_rel": float(np.nanmax(rel)) if a.size else 0.0,
        "all_exact": bool(exact == a.size),
    }


CASES = [
    ("c1", {"shape": [3, 3], "singular": True}, 1730),
    ("c3", {"shape": [2, 7, 7], "singular": True}, 1732),
]

report = {"cases": {}}

for name, spec, seed in CASES:
    A = build(spec, seed)
    n = A.shape[-1]
    Aflat = A.reshape(-1, n, n)
    batch = Aflat.shape[0]
    ref_inv, ref_info = torch.linalg.inv_ex(A, check_errors=False)
    I = torch.eye(n, dtype=A.dtype).expand(Aflat.shape[0], n, n).contiguous()

    entry = {"ref_info": ref_info.reshape(-1).tolist(), "ref_inv": ref_inv.reshape(batch, n, n)[0].tolist()}

    lu_t, piv_t, info_t = torch.linalg.lu_factor_ex(Aflat)
    cand = {}
    calls = {
        "inv": lambda: torch.linalg.inv(A),
        "solve_ex": lambda: torch.linalg.solve_ex(Aflat, I)[0],
        "solve": lambda: torch.linalg.solve(Aflat, I),
        "lu_solve_flat": lambda: torch.linalg.lu_solve(lu_t, piv_t, I),
        "lu_solve_batched_arg": lambda: torch.linalg.lu_solve(
            *torch.linalg.lu_factor(Aflat), I
        ),
        "pinv": lambda: torch.linalg.pinv(A),
    }
    for k, fn in calls.items():
        try:
            v = fn().reshape(batch, n, n)
            cand[k] = bits_eq(v, ref_inv.reshape(batch, n, n))
        except Exception as exc:
            cand[k] = {"error": repr(exc)[:160]}
    entry["torch_callables"] = cand

    # (b)/(c) emulations
    emu = {}
    for uname, upd in UPD.items():
        # LU on A itself
        for b in range(batch):
            Ab = Aflat[b].numpy()
            M, piv, info = emu_lu(Ab, upd)
            X = emu_solve(M, piv, upd)
            emu.setdefault(f"lu_A_{uname}", []).append(
                bits_eq(X, ref_inv.reshape(batch, n, n)[b].numpy())
            )
            # LU on the logical transpose
            Mt, pivt, infot = emu_lu(Ab.T.copy(), upd)
            Xt = emu_solve(Mt, pivt, upd)
            emu.setdefault(f"lu_AT_{uname}", []).append(
                bits_eq(Xt.T, ref_inv.reshape(batch, n, n)[b].numpy())
            )
            for norm in ("div", "recip"):
                G = emu_gj(Ab, upd, norm)
                emu.setdefault(f"gj_{norm}_{uname}", []).append(
                    bits_eq(G, ref_inv.reshape(batch, n, n)[b].numpy())
                )
    entry["emulations"] = {
        k: {"all_exact": all(r.get("all_exact", False) for r in v),
            "exact": sum(r.get("exact", 0) for r in v),
            "numel": sum(r.get("numel", 0) for r in v),
            "per_matrix": v}
        for k, v in emu.items()
    }

    # (d) implied tiny pivot if the reference were LU-based on A
    M, piv, info = emu_lu(Aflat[0].numpy(), upd_fma)
    Xref = ref_inv.reshape(batch, n, n)[0].numpy().astype(np.float64)
    L = np.tril(M.astype(np.float64), -1) + np.eye(n)
    U = np.triu(M.astype(np.float64))
    implied = []
    for k in range(n):
        b = np.zeros(n)
        b[k] = 1.0
        for j in range(n):
            p = int(piv[j])
            if p != j:
                b[j], b[p] = b[p], b[j]
        y = np.linalg.solve(L, b)
        # last component goes through u22
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = y[n - 1] / Xref[n - 1, k]
        implied.append(
            {
                "k": k,
                "y_last": float(y[n - 1]),
                "ref_x_last": float(Xref[n - 1, k]),
                "implied_u22": float(delta),
                "predicted_x": np.linalg.solve(U + np.diag([0.0] * (n - 1) + [0.0]), y).tolist()
                if False
                else None,
            }
        )
    entry["implied_pivot"] = {
        "y": implied,
        "u_diag": np.diag(M).tolist(),
        "pivots": piv.tolist(),
        "L": L[:, :3].tolist() if n == 3 else None,
        "U": U.tolist() if n == 3 else None,
    }
    report["cases"][name] = entry

with open(os.path.join(OUT, "probe2.json"), "w") as f:
    json.dump(report, f, indent=1)
print(json.dumps(report["cases"]["c1"], indent=1)[:7000])
