"""Diagnostic: identify the exact path/arithmetic of the target reference inverse.

Two questions:
 1) which matrix is factorized (A or A^T) -- this decides whether an exactly
    singular input yields an exact-zero pivot (info>0) or a tiny residual pivot
    (info==0, huge finite inverse values like the reference);
 2) which solve ordering reproduces the reference values.
"""
import json
import os
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")


def gen_inputs(spec, seed, device="cpu"):
    """Faithful copy of the frozen reference generator (two randn draws!)."""
    dtype = getattr(torch, spec["dtype"])
    gen = torch.Generator().manual_seed(seed)

    def random(shape, positive=False):
        x = torch.randn(shape, generator=gen)
        if positive:
            x = x.abs() + 0.05
        return x.to(device=device, dtype=dtype)

    _unused = random(spec["shape"])          # first draw is discarded by the reference
    x = random(spec["shape"])
    x = x + torch.eye(x.shape[-1], device=device, dtype=dtype) * (x.shape[-1] + 1)
    if spec.get("singular"):
        x[..., -1, :] = x[..., 0, :]
    return x


def emu_lu(a, mult="rcp", upd="mulsub"):
    work = np.array(a, dtype=np.float32, copy=True)
    n = work.shape[0]
    piv, info = [], 0
    for j in range(n):
        p = int(np.argmax(np.abs(work[j:, j]))) + j
        piv.append(p)
        if p != j:
            work[[j, p], :] = work[[p, j], :]
        pv = work[j, j]
        if pv != 0.0:
            l = (work[j + 1:, j] / pv).astype(np.float32) if mult == "div" else \
                (work[j + 1:, j] * (np.float32(1.0) / pv)).astype(np.float32)
            work[j + 1:, j] = l
            if j + 1 < n:
                r0, sub = work[j, j + 1:], work[j + 1:, j + 1:]
                if upd == "mulsub":
                    work[j + 1:, j + 1:] = (sub - l[:, None] * r0[None, :]).astype(np.float32)
                else:
                    work[j + 1:, j + 1:] = (sub.astype(np.float64) -
                                            l[:, None].astype(np.float64) *
                                            r0[None, :].astype(np.float64)).astype(np.float32)
        elif info == 0:
            info = j + 1
    return work, piv, info


def solve(work, piv, rhs, upd="mulsub", recipdiag=False, perm="apply", order="fwd_bwd"):
    n = work.shape[0]
    b = np.array(rhs, dtype=np.float32, copy=True)
    if perm == "apply":
        for j, p in enumerate(piv):
            if p != j:
                b[[j, p], :] = b[[p, j], :]
    elif perm == "reverse":
        for j in range(n - 1, -1, -1):
            p = piv[j]
            if p != j:
                b[[j, p], :] = b[[p, j], :]
    def upd_rows(dst, src, fac):
        if upd == "mulsub":
            return (dst - fac[:, None] * src[None, :]).astype(np.float32)
        return (dst.astype(np.float64) - fac[:, None].astype(np.float64) *
                src[None, :].astype(np.float64)).astype(np.float32)
    if order == "fwd_bwd":
        for j in range(n):
            if j + 1 < n:
                b[j + 1:, :] = upd_rows(b[j + 1:, :], b[j], work[j + 1:, j])
        for j in range(n - 1, -1, -1):
            d = work[j, j]
            b[j, :] = (b[j, :] * (np.float32(1.0) / d)).astype(np.float32) if recipdiag \
                else (b[j, :] / d).astype(np.float32)
            if j > 0:
                b[:j, :] = upd_rows(b[:j, :], b[j], work[:j, j])
    else:  # adjoint-style: (L U)^-T x = b  -> U^T y = P b, L^T x = y
        for j in range(n):
            if j + 1 < n:
                b[j + 1:, :] = upd_rows(b[j + 1:, :], b[j], work[j, j + 1:])
        for j in range(n - 1, -1, -1):
            d = work[j, j]
            b[j, :] = (b[j, :] / d).astype(np.float32)
            if j > 0:
                b[:j, :] = upd_rows(b[:j, :], b[j], work[j + 1:, j][: j][::-1] if False else work[1:, j][:j])
    return b


def cmp(ref, cand, name):
    ref = np.asarray(ref, dtype=np.float32)
    cand = np.asarray(cand, dtype=np.float32)
    ok = np.isclose(ref, cand, rtol=1e-4, atol=1e-4, equal_nan=True)
    with np.errstate(invalid="ignore"):
        rel = np.abs(cand - ref) / np.maximum(np.abs(ref), np.float32(1e-30))
    return {"variant": name, "allclose": bool(ok.all()), "frac": float(ok.mean()),
            "max_rel": float(np.nanmax(rel)) if np.isfinite(rel).any() else None,
            "bit_exact": bool(np.array_equal(ref, cand))}


cases = [
    ("c0_reg3", {"dtype": "float32", "shape": [3, 3], "singular": False}, 1729),
    ("c1_sing3", {"dtype": "float32", "shape": [3, 3], "singular": True}, 1730),
    ("c2_reg7", {"dtype": "float32", "shape": [2, 7, 7], "singular": False}, 1731),
    ("c3_sing7", {"dtype": "float32", "shape": [2, 7, 7], "singular": True}, 1732),
]

out = {"cuda_or_cpu": str(torch.__version__), "cases": {}}
for name, spec, seed in cases:
    A = gen_inputs(spec, seed)
    n = spec["shape"][-1]
    ref = torch.linalg.inv_ex(A, check_errors=False)
    Af = A.reshape(-1, n, n)
    rec = {"A": Af.tolist(), "ref_inverse": ref.inverse.reshape(-1, n, n).tolist(),
           "ref_info": ref.info.tolist(), "rows0_last_equal": bool(torch.equal(Af[..., 0, :], Af[..., -1, :])),
           "batches": []}
    for b in range(Af.shape[0]):
        a = np.ascontiguousarray(Af[b].numpy())
        refinv = ref.inverse.reshape(-1, n, n)[b].numpy()
        e = {"b": b}
        for tag, M in (("AT", np.ascontiguousarray(a.T)), ("A", a)):
            LU_t, piv_t, info_t = torch.linalg.lu_factor_ex(torch.from_numpy(np.ascontiguousarray(M)))
            LU_t = LU_t.numpy()
            piv_list = (piv_t.reshape(-1, piv_t.shape[-1])[0].numpy() - 1).tolist()
            e[f"{tag}_lu_info"] = int(info_t.reshape(-1)[0])
            e[f"{tag}_lu_diag"] = LU_t.diagonal().tolist()
            e[f"{tag}_pivots"] = piv_list
            em = {}
            for mult in ("rcp", "div"):
                for upd in ("mulsub", "fma"):
                    wk, piv, info = emu_lu(M, mult, upd)
                    em[f"{mult}/{upd}"] = {
                        "info": info,
                        "bit_exact_vs_torch": bool(np.array_equal(wk, LU_t)),
                        "pivots_equal": piv == piv_list,
                        "max_diff": float(np.abs(wk - LU_t).max()),
                    }
            e[f"{tag}_lu_emulation"] = em
            # torch-level solve on torch's own factors, then compare to reference
            eye = torch.eye(n)
            y = torch.linalg.lu_solve(torch.from_numpy(LU_t), torch.from_numpy(
                np.asarray(piv_list, dtype=np.int64) + 1), eye)
            cand = y.numpy().T if tag == "AT" else y.numpy()
            e[f"{tag}_lu_solve_cmp"] = cmp(refinv, cand, f"torch_lu_solve_on_{tag}")
            e[f"{tag}_lu_solve_vals"] = cand.tolist()
            # emulated solves on emulated factors
            for mult in ("rcp", "div"):
                for upd in ("mulsub", "fma"):
                    wk, piv, info = emu_lu(M, mult, upd)
                    for recip in (False, True):
                        s = solve(wk, piv, np.eye(n, dtype=np.float32), upd, recip)
                        cand2 = s.T if tag == "AT" else s
                        e.setdefault("emulated_solve_cmp", []).append(
                            cmp(refinv, np.ascontiguousarray(cand2), f"{tag}/{mult}/{upd}/recip={recip}"))
            # torch-level ops for path identification
            for opname, val in (
                ("inv", torch.linalg.inv(Af[b])),
                ("solve_ex", torch.linalg.solve_ex(Af[b], torch.eye(n))[0]),
                ("inverse", torch.inverse(Af[b])),
            ):
                e.setdefault("torch_ops_cmp", []).append(cmp(refinv, val.numpy(), opname))
        rec["batches"].append(e)
    out["cases"][name] = rec

with open(os.path.join(ART, "probe_path.json"), "w") as fh:
    json.dump(out, fh, indent=1)

brief = {}
for name, rec in out["cases"].items():
    rows = []
    for b in rec["batches"]:
        # verify our input generation against the frozen reference values
        rows.append({
            "b": b["b"],
            "A_lu_info": b["A_lu_info"], "AT_lu_info": b["AT_lu_info"],
            "A_diag_last": b["A_lu_diag"][-1], "AT_diag_last": b["AT_lu_diag"][-1],
            "AT_lu_emulation": {k: (v["info"], v["bit_exact_vs_torch"]) for k, v in b["AT_lu_emulation"].items()},
            "A_lu_emulation": {k: (v["info"], v["bit_exact_vs_torch"]) for k, v in b["A_lu_emulation"].items()},
            "torch_AT_lu_solve": [b["AT_lu_solve_cmp"]["allclose"], b["AT_lu_solve_cmp"]["max_rel"], b["AT_lu_solve_cmp"]["bit_exact"]],
            "torch_A_lu_solve": [b["A_lu_solve_cmp"]["allclose"], b["A_lu_solve_cmp"]["max_rel"], b["A_lu_solve_cmp"]["bit_exact"]],
            "emul_best": sorted([(v["allclose"], v["max_rel"] if v["max_rel"] is not None else 1e9, v["variant"])
                                 for v in b.get("emulated_solve_cmp", [])])[:3],
            "torch_ops": {v["variant"]: (v["allclose"], v["bit_exact"]) for v in b.get("torch_ops_cmp", [])},
        })
    brief[name] = {"ref_info": rec["ref_info"], "rows_equal": rec["rows0_last_equal"], "batches": rows}
print(json.dumps(brief, indent=1)[:12000])
