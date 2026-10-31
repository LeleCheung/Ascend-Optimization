"""Diagnostic: identify which torch operation the reference inverse equals.

Inputs are the validated frozen ones (two randn draws). If the reference equals,
bit for bit, some composition of public torch ops, that pins down the exact
arithmetic path (and gives us the exact LU factors to replicate).
"""
import json
import os
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")


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


def cmp(ref, cand):
    ref = np.asarray(ref, dtype=np.float32)
    cand = np.asarray(cand, dtype=np.float32)
    ok = np.isclose(ref, cand, rtol=1e-4, atol=1e-4, equal_nan=True)
    with np.errstate(invalid="ignore"):
        rel = np.abs(cand - ref) / np.maximum(np.abs(ref), np.float32(1e-30))
    return {"allclose": bool(ok.all()), "bad": int((~ok).sum()),
            "bit_exact": bool(np.array_equal(ref, cand)),
            "max_rel": float(np.nanmax(rel)) if np.isfinite(rel).any() else None,
            "nan_in_cand": bool((~np.isfinite(cand)).any())}


cases = [
    ("c1_sing3", {"dtype": "float32", "shape": [3, 3], "singular": True}, 1730),
    ("c3_sing7", {"dtype": "float32", "shape": [2, 7, 7], "singular": True}, 1732),
    ("c0_reg3", {"dtype": "float32", "shape": [3, 3], "singular": False}, 1729),
]

lines, out = [], {"cases": {}}
for name, spec, seed in cases:
    A = gen_inputs(spec, seed)
    n = spec["shape"][-1]
    Af = A.reshape(-1, n, n)
    ref = torch.linalg.inv_ex(A, check_errors=False)
    refi = ref.inverse.reshape(-1, n, n)
    rec = {"ref_info": ref.info.tolist(), "batches": []}
    for b in range(Af.shape[0]):
        Ab = Af[b]
        r = refi[b].numpy()
        e = {"b": b}
        I = torch.eye(n)
        ops = {}
        # candidate compositions of public ops
        for tag, AT in (("A", False), ("AT", True)):
            M = Ab.t() if AT else Ab
            LU, piv, info = torch.linalg.lu_factor_ex(M)
            e[f"{tag}_lu_info"] = int(info.reshape(-1)[0])
            e[f"{tag}_lu_factors"] = LU.numpy().tolist()
            e[f"{tag}_lu_pivots"] = (piv.reshape(-1) - 1).tolist()
            for mode in ("plain", "adj"):
                try:
                    y = torch.linalg.lu_solve(LU, piv, I, left=True, adjoint=(mode == "adj"))
                except Exception as exc:  # noqa: BLE001
                    ops[f"lu_solve_{tag}_{mode}"] = {"error": repr(exc)}
                    continue
                for tr in (False, True):
                    cand = y.t().numpy() if tr else y.numpy()
                    ops[f"lu_solve_{tag}_{mode}" + ("_T" if tr else "")] = cmp(r, cand)
        for tag, fn in (
            ("inv", lambda: torch.linalg.inv(Ab)),
            ("inverse", lambda: torch.inverse(Ab)),
            ("solve_ex", lambda: torch.linalg.solve_ex(Ab, I)[0]),
            ("solve", lambda: torch.linalg.solve(Ab, I)),
            ("lu_factor_then_lu_solve", lambda: torch.linalg.lu_solve(
                *torch.linalg.lu_factor_ex(Ab)[:2], I)),
            ("pinv", lambda: torch.linalg.pinv(Ab)),
        ):
            try:
                ops[tag] = cmp(r, fn().numpy())
            except Exception as exc:  # noqa: BLE001
                ops[tag] = {"error": repr(exc)}
        e["ops"] = ops
        rec["batches"].append(e)
        lines.append(f"{name}[{b}] lu_info A={e['A_lu_info']} AT={e['AT_lu_info']} "
                     f"pivA={e['A_lu_pivots']} pivAT={e['AT_lu_pivots']}")
        for k, v in ops.items():
            if "error" in v:
                lines.append(f"   {k}: ERROR {v['error'][:60]}")
            else:
                lines.append(f"   {k}: ok={v['allclose']} bit={v['bit_exact']} "
                             f"bad={v['bad']} maxrel={v['max_rel']} nan={v['nan_in_cand']}")
    out["cases"][name] = rec

with open(os.path.join(ART, "probe_ingredients.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print("\n".join(lines))
