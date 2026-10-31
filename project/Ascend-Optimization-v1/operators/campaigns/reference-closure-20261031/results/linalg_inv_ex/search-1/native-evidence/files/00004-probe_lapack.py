"""Identify the exact arithmetic of the reference linalg_inv_ex path (diagnostic only).

Corrected input reconstruction: gen_inputs calls random() twice for
linalg_inv_ex, so the first randn draw is discarded and the second is used.

Validated against the frozen audit values for linalg_inv_ex-correctness-1
(npu_reference inverse row0 = [-7622158.0, 0.019524509087204933, 7622158.0],
 info = 0).
"""

import json
import os

import numpy as np
import torch

OUT = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")


def build(spec, seed):
    n = spec["shape"][-1]
    g = torch.Generator().manual_seed(seed)
    _discarded = torch.randn(spec["shape"], generator=g)  # first random() call
    x = torch.randn(spec["shape"], generator=g)  # second random() call
    x = x + torch.eye(n) * (n + 1)
    if spec.get("singular"):
        x[..., -1, :] = x[..., 0, :]
    return x.to(torch.float32).contiguous()


def upd_fma(m, y, cur):
    """cur - m*y with a single rounding (fma)."""
    return np.float32(np.float64(cur) - np.float64(m) * np.float64(y))


def upd_mul(m, y, cur):
    return np.float32(cur - np.float32(m * y))


UPD = {"fma": upd_fma, "mul": upd_mul}


def emu_lu(A, upd, scale):
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


def bits_eq(a, b):
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    b = np.ascontiguousarray(np.asarray(b, dtype=np.float32))
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    exact = int(np.sum(a.view(np.int32) == b.view(np.int32)))
    d = np.abs(np.float64(a) - np.float64(b))
    rel = np.where(np.float64(np.abs(b)) > 0, d / np.abs(np.float64(b)), d)
    return {
        "exact": exact,
        "numel": int(a.size),
        "max_abs": float(np.nanmax(d)) if a.size else 0.0,
        "max_rel": float(np.nanmax(rel)) if a.size else 0.0,
        "all_exact": bool(exact == a.size),
    }


CASES = [
    ("correctness-1", {"dtype": "float32", "shape": [3, 3], "singular": True}, 1730),
    ("correctness-0", {"dtype": "float32", "shape": [3, 3], "singular": False}, 1729),
    ("correctness-3", {"dtype": "float32", "shape": [2, 7, 7], "singular": True}, 1732),
]

report = {"torch_version": torch.__version__, "numpy_version": np.__version__, "cases": {}}

try:
    import torch_npu  # noqa: F401

    report["torch_npu"] = torch_npu.__version__
    has_npu = torch.npu.is_available()
except Exception as exc:  # pragma: no cover
    report["torch_npu_error"] = repr(exc)
    has_npu = False
report["has_npu"] = has_npu

for name, spec, seed in CASES:
    A = build(spec, seed)
    n = A.shape[-1]
    Aflat = A.reshape(-1, n, n)
    batch = Aflat.shape[0]

    ref_inv, ref_info = torch.linalg.inv_ex(A, check_errors=False)
    entry = {
        "shape": list(A.shape),
        "A_row0": A.reshape(batch, n, n)[0][0].tolist(),
        "A_rowlast": A.reshape(batch, n, n)[0][-1].tolist(),
        "rows_identical_bitwise": bool(
            torch.equal(A.reshape(batch, n, n)[0][0], A.reshape(batch, n, n)[0][-1])
        ),
        "cpu_ref_info": ref_info.reshape(-1).tolist(),
        "cpu_ref_inv_row0": ref_inv.reshape(batch, n, n)[0][0].tolist(),
        "cpu_ref_inv_absmax": float(ref_inv.abs().max()),
    }

    lu, pivs, infos_ex = torch.linalg.lu_factor_ex(A)
    entry["lu_factor_ex_info"] = infos_ex.reshape(-1).tolist()
    entry["lu_diag"] = [
        lu.reshape(batch, n, n)[b].diagonal().tolist() for b in range(batch)
    ]

    if has_npu:
        try:
            An = A.to("npu")
            inv_npu, info_npu = torch.linalg.inv_ex(An, check_errors=False)
            entry["npu_ref_info"] = info_npu.cpu().reshape(-1).tolist()
            entry["npu_vs_cpu_inv"] = bits_eq(inv_npu.cpu(), ref_inv)
        except Exception as exc:
            entry["npu_error"] = repr(exc)

    entry["variants"] = {}
    for scale in ("recip", "div"):
        for uname, upd in UPD.items():
            lu_bits = lu_total = piv_ok = inv_bits = inv_tot = 0
            inv_max_rel = 0.0
            infos = []
            for b in range(batch):
                Ab = Aflat[b].numpy()
                M, piv, info = emu_lu(Ab, upd, scale)
                infos.append(info)
                r = bits_eq(M, lu.reshape(-1, n, n)[b].numpy())
                lu_bits += r["exact"]
                lu_total += r["numel"]
                piv_ok += int(np.array_equal(piv, pivs.reshape(-1, n)[b].numpy()))
                X = emu_solve(M, piv, upd)
                ri = bits_eq(X, ref_inv.reshape(-1, n, n)[b].numpy())
                inv_bits += ri["exact"]
                inv_tot += ri["numel"]
                inv_max_rel = max(inv_max_rel, ri["max_rel"])
            entry["variants"][f"{scale}_{uname}"] = {
                "lu_exact": lu_bits,
                "lu_numel": lu_total,
                "pivots_match": piv_ok,
                "inv_exact": inv_bits,
                "inv_numel": inv_tot,
                "inv_max_rel": inv_max_rel,
                "emu_info": infos,
            }
    report["cases"][name] = entry

with open(os.path.join(OUT, "probe.json"), "w") as f:
    json.dump(report, f, indent=1)
print(json.dumps(report, indent=1)[:9000])
