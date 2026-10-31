"""Identify the exact arithmetic of the reference linalg_inv_ex path (diagnostic only).

The reference on this target is torch.linalg.inv_ex routed to CPU LAPACK
(values are bit-identical to the CPU observation).  For the two singular
workloads the reference reports info=0 and finite ~1e8 entries, which can only
happen if the elimination does NOT cancel identical rows exactly.  The most
likely cause is that LAPACK's dgetf2 scales with a rounded reciprocal
(DSCAL with ONE/A(j,j)) instead of dividing element-wise.

This script emulates candidate LU + solve arithmetics in float32 (with an
exact fma emulation via float64) and reports bit-exact agreement with
torch's own lu_factor / inv_ex.
"""

import json
import os

import numpy as np
import torch

OUT = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")


def build(spec, seed):
    n = spec["shape"][-1]
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(spec["shape"], generator=g)
    x = x + torch.eye(n) * (n + 1)
    if spec.get("singular"):
        x[..., -1, :] = x[..., 0, :]
    return x.to(torch.float32).contiguous()


# ---- float32 primitive with selectable rounding -------------------------
def upd_fma(m, y, cur):
    """cur - m*y with a single rounding (fma)."""
    return np.float32(np.float64(cur) - np.float64(m) * np.float64(y))


def upd_mul(m, y, cur):
    """cur - m*y with two roundings."""
    return np.float32(cur - np.float32(m * y))


UPD = {"fma": upd_fma, "mul": upd_mul}


def emu_lu(A, upd, scale):
    """Partial-pivoting LU.  scale in {'recip','div'}."""
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
                    piv_val = np.float32(M[j, j])
                    M[j + 1:, j] = np.float32(M[j + 1:, j] / piv_val)
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


def bits_eq(a, b):
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    b = np.ascontiguousarray(np.asarray(b, dtype=np.float32))
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    exact = int(np.sum(a.view(np.int32) == b.view(np.int32)))
    nan_bad = ~(np.isnan(a) == np.isnan(b))
    d = np.abs(np.float64(a) - np.float64(b))
    rel = np.where(np.float64(np.abs(b)) > 0, d / np.abs(np.float64(b)), d)
    return {
        "exact": exact,
        "numel": int(a.size),
        "max_abs": float(np.nanmax(d)) if a.size else 0.0,
        "max_rel": float(np.nanmax(rel)) if a.size else 0.0,
        "nan_mismatch": int(np.sum(nan_bad)),
        "all_exact": bool(exact == a.size),
    }


CASES = [
    ("correctness-0", {"dtype": "float32", "shape": [3, 3], "singular": False}, 1729),
    ("correctness-1", {"dtype": "float32", "shape": [3, 3], "singular": True}, 1730),
    ("correctness-2", {"dtype": "float32", "shape": [2, 7, 7], "singular": False}, 1731),
    ("correctness-3", {"dtype": "float32", "shape": [2, 7, 7], "singular": True}, 1732),
    ("timing-0", {"dtype": "float32", "shape": [16, 16]}, 1729),
    ("timing-1", {"dtype": "float32", "shape": [8, 32, 32]}, 1730),
]

report = {
    "torch_version": torch.__version__,
    "numpy_version": np.__version__,
    "cpu": torch.__config__.parallel_info(),
    "cases": {},
}

for name, spec, seed in CASES:
    A = build(spec, seed)
    n = A.shape[-1]
    Aflat = A.reshape(-1, n, n)
    batch = Aflat.shape[0]

    ref_inv, ref_info = torch.linalg.inv_ex(A, check_errors=False)
    lu, piv_t = torch.linalg.lu_factor(A)
    eye = torch.eye(n).expand_as(Aflat).contiguous()
    lu_solve_res = torch.linalg.lu_solve(lu, piv_t, eye)
    inv_plain = torch.linalg.inv(A)

    entry = {
        "shape": list(A.shape),
        "ref_info": ref_info.reshape(-1).tolist(),
        "ref_inv_flat0": ref_inv.reshape(-1)[:9].tolist(),
        "ref_absmax": float(ref_inv.abs().max()),
        "lu_vs_lu_solve": bits_eq(lu_solve_res.reshape_as(ref_inv), ref_inv),
        "inv_vs_inv_ex": bits_eq(inv_plain, ref_inv),
        "variants": {},
    }

    for scale in ("recip", "div"):
        for uname, upd in UPD.items():
            lu_bits = 0
            lu_total = 0
            piv_ok = 0
            inv_bits = 0
            inv_total = 0
            inv_max_rel = 0.0
            infos = []
            lu_exact_this = 0
            for b in range(batch):
                Ab = Aflat[b].numpy()
                M, piv, info = emu_lu(Ab, upd, scale)
                infos.append(info)
                lu_ref = lu.reshape(-1, n, n)[b].numpy()
                r = bits_eq(M, lu_ref)
                lu_bits += r["exact"]
                lu_total += r["numel"]
                lu_exact_this += int(r["all_exact"])
                piv_ok += int(np.array_equal(piv, piv_t.reshape(-1, n)[b].numpy()))
                X = emu_solve(M, piv, upd)
                ref_b = ref_inv.reshape(-1, n, n)[b].numpy()
                ri = bits_eq(X, ref_b)
                inv_bits += ri["exact"]
                inv_total += ri["numel"]
                inv_max_rel = max(inv_max_rel, ri["max_rel"])
            entry["variants"][f"{scale}_{uname}"] = {
                "lu_exact": lu_bits,
                "lu_numel": lu_total,
                "lu_all_matrices_exact": lu_exact_this,
                "pivots_match": piv_ok,
                "inv_exact": inv_bits,
                "inv_numel": inv_total,
                "inv_max_rel": inv_max_rel,
                "emu_info": infos,
            }
    report["cases"][name] = entry

with open(os.path.join(OUT, "probe.json"), "w") as f:
    json.dump(report, f, indent=1)
print(json.dumps(report, indent=1)[:6000])
