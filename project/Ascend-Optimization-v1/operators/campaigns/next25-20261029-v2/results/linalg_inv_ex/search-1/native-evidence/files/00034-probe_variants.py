"""Diagnostic: identify the exact reference arithmetic, and locate the current
candidate's non-finite output.

Inputs are reproduced with the frozen generator's two randn draws; they are
validated by checking that torch.linalg.inv_ex on the target reproduces the
recorded contract-probe reference values. Then a grid of LU/solve variants is
compared against that reference, and the current candidate kernel output is
dumped for the singular 3x3.
"""
import json
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")

SPECS = {
    "c0_reg3": ({"dtype": "float32", "shape": [3, 3], "singular": False}, 1729),
    "c1_sing3": ({"dtype": "float32", "shape": [3, 3], "singular": True}, 1730),
    "c2_reg7": ({"dtype": "float32", "shape": [2, 7, 7], "singular": False}, 1731),
    "c3_sing7": ({"dtype": "float32", "shape": [2, 7, 7], "singular": True}, 1732),
}

# frozen reference values recorded by the campaign contract probe
FROZEN = {
    "c0_reg3": [[0.347623735666275, -0.25079676508903503, -0.20276117324829102],
                [-0.04192081838846207, 0.28018462657928467, 0.048222802579402924],
                [-0.13759201765060425, 0.18680976331233978, 0.3735128343105316]],
    "c1_sing3": [[-7622158.0, 0.019524509087204933, 7622158.0],
                 [-18357816.0, 0.3507829010486603, 18357816.0],
                 [105041440.0, -0.709788978099823, -105041440.0]],
}


def gen_inputs(spec, seed):
    dtype = getattr(torch, spec["dtype"])
    gen = torch.Generator().manual_seed(seed)

    def random(shape):
        return torch.randn(shape, generator=gen).to(dtype=dtype)

    random(spec["shape"])                      # discarded first draw (frozen gen)
    x = random(spec["shape"])
    x = x + torch.eye(x.shape[-1], dtype=dtype) * (x.shape[-1] + 1)
    if spec.get("singular"):
        x[..., -1, :] = x[..., 0, :]
    return x


def emu_lu(a, mult, upd):
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
        else:
            l = (work[j + 1:, j] * (np.float32(1.0) / pv)).astype(np.float32)  # inf, as LAPACK
            if info == 0:
                info = j + 1
        work[j + 1:, j] = l
        if j + 1 < n:
            r0, sub = work[j, j + 1:], work[j + 1:, j + 1:]
            if upd == "mulsub":
                work[j + 1:, j + 1:] = (sub - l[:, None] * r0[None, :]).astype(np.float32)
            else:
                work[j + 1:, j + 1:] = (sub.astype(np.float64) -
                                        l[:, None].astype(np.float64) *
                                        r0[None, :].astype(np.float64)).astype(np.float32)
    return work, piv, info


def upd_rows(dst, src, c, upd):
    if upd == "mulsub":
        return (dst - c[:, None] * src[None, :]).astype(np.float32)
    return (dst.astype(np.float64) - c[:, None].astype(np.float64) *
            src[None, :].astype(np.float64)).astype(np.float32)


def emu_solve(work, piv, n, upd, order):
    b = np.eye(n, dtype=np.float32)
    for j, p in enumerate(piv):
        if p != j:
            b[[j, p], :] = b[[p, j], :]
    if order == "N":
        for j in range(n):
            if j + 1 < n:
                b[j + 1:, :] = upd_rows(b[j + 1:, :], b[j], work[j + 1:, j], upd)
        for j in range(n - 1, -1, -1):
            b[j, :] = (b[j, :] / work[j, j]).astype(np.float32)
            if j > 0:
                b[:j, :] = upd_rows(b[:j, :], b[j], work[:j, j], upd)
    else:  # trans='T': U^T forward then L^T back, both via row j of the factors
        for j in range(n):
            b[j, :] = (b[j, :] / work[j, j]).astype(np.float32)
            if j + 1 < n:
                b[j + 1:, :] = upd_rows(b[j + 1:, :], b[j], work[j, j + 1:], upd)
        for j in range(n - 1, -1, -1):
            if j > 0:
                b[:j, :] = upd_rows(b[:j, :], b[j], work[j, :j], upd)
    return b


def stat(ref, cand):
    ref = np.asarray(ref, dtype=np.float32)
    cand = np.asarray(cand, dtype=np.float32)
    fin = np.isfinite(cand)
    ok = np.isclose(ref, cand, rtol=1e-4, atol=1e-4, equal_nan=True)
    with np.errstate(invalid="ignore"):
        rel = np.abs(cand - ref) / np.maximum(np.abs(ref), np.float32(1e-30))
    return (bool(ok.all()), int((~ok).sum()), int((~fin).sum()),
            float(np.nanmax(rel)) if np.isfinite(rel).any() else -1.0,
            bool(np.array_equal(ref, cand)))


out = {"env": {"torch": torch.__version__}, "cases": {}}
lines = []

for name, (spec, seed) in SPECS.items():
    A = gen_inputs(spec, seed)
    n = spec["shape"][-1]
    Af = A.reshape(-1, n, n)
    ref = torch.linalg.inv_ex(A, check_errors=False)
    refi = ref.inverse.reshape(-1, n, n).numpy()
    rec = {"A": Af.tolist(), "ref_info": ref.info.tolist(),
           "ref_min": float(refi.min()), "ref_max": float(refi.max()), "batches": []}
    if name in FROZEN:
        rec["input_validated"] = bool(np.array_equal(refi[0], np.asarray(FROZEN[name], dtype=np.float32)))
        lines.append(f"{name}: input reproduction valid={rec['input_validated']} info={rec['ref_info']}")
    else:
        lines.append(f"{name}: ref_min={rec['ref_min']:.6g} ref_max={rec['ref_max']:.6g} info={rec['ref_info']}")
    for b in range(Af.shape[0]):
        a = np.ascontiguousarray(Af[b].numpy())
        r = refi[b]
        hits = []
        for mtag, M in (("A", a), ("AT", np.ascontiguousarray(a.T))):
            for mult in ("div", "rcp"):
                for upd in ("mulsub", "fma"):
                    wk, piv, info = emu_lu(M, mult, upd)
                    for order in ("N", "T"):
                        z = emu_solve(wk, piv, n, upd, order)
                        for trans in (False, True):
                            cand = np.ascontiguousarray(z.T) if trans else z
                            ok, nbad, nnf, maxrel, exact = stat(r, cand)
                            if ok:
                                hits.append(f"{mtag}/{mult}/{upd}/{order}/res{'^T' if trans else ''}"
                                            f" info={info} exact={exact} maxrel={maxrel:.3g}")
        rec["batches"].append({"b": b, "hits": hits})
        lines.append(f"  {name}[{b}] matching variants: {hits if hits else 'NONE'}")
    out["cases"][name] = rec

# dump the current candidate kernel output for the singular 3x3
try:
    sys.path.insert(0, os.getcwd())
    import main as cand_mod
    A1 = gen_inputs(*SPECS["c1_sing3"])
    got = cand_mod.run(A1, check_errors=False)
    gi = got.inverse.numpy()
    out["candidate_c1"] = {"inverse": gi.tolist(), "info": got.info.tolist(),
                           "finite": bool(np.isfinite(gi).all())}
    lines.append("candidate c1_sing3 inverse=" + np.array2string(gi, precision=6) +
                 f" info={got.info.tolist()} finite={np.isfinite(gi).all()}")
except Exception as exc:  # noqa: BLE001
    out["candidate_c1"] = {"error": repr(exc)}
    lines.append("candidate import/run failed: " + repr(exc))

with open(os.path.join(ART, "probe_variants.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print("\n".join(lines))
