"""Diagnostic: understand target reference behaviour for linalg_inv_ex singular cases.

Does NOT patch environment, does not implement the candidate. Only inspects what
torch.linalg.inv_ex does on the frozen workload inputs and which scalar LU
variant reproduces its rounding on the singular (info==0, huge finite) cases.
"""
import json
import os
import sys

import numpy as np
import torch

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")


def gen_inputs(spec, seed, device):
    dtype = getattr(torch, spec["dtype"])
    gen = torch.Generator().manual_seed(seed)

    def random(shape, positive=False):
        x = torch.randn(shape, generator=gen)
        if positive:
            x = x.abs() + 0.05
        return x.to(device=device, dtype=dtype)

    x = random(spec["shape"])
    x = x + torch.eye(x.shape[-1], device=device, dtype=dtype) * (x.shape[-1] + 1)
    if spec.get("singular"):
        x[..., -1, :] = x[..., 0, :]
    return x


def lu_inv(a, trans_first=False):
    """Scalar fp32 partial-pivot LU + solve, mimicking LAPACK sgetf2 ordering.

    trans_first=True  -> factor the column-major image of a row-major matrix,
    i.e. LU of a.T (this is what ATen hands to LAPACK).
    Returns (inv, info, pivots, last_diag).
    """
    m = a.T.astype(np.float32) if trans_first else a.astype(np.float32)
    n = m.shape[0]
    work = np.array(m, dtype=np.float32, copy=True)
    piv = []
    info = 0
    for j in range(n):
        col = np.abs(work[j:, j])
        p = int(np.argmax(col)) + j
        piv.append(p)
        if p != j:
            tmp = work[j].copy()
            work[j] = work[p]
            work[p] = tmp
        if work[j, j] != 0:
            l = (work[j + 1:, j] / work[j, j]).astype(np.float32)
            work[j + 1:, j] = l
            if j + 1 < n:
                work[j + 1:, j + 1:] = (work[j + 1:, j + 1:] -
                                        l[:, None] * work[j, j + 1:][None, :]).astype(np.float32)
        else:
            if info == 0:
                info = j + 1
    last_diag = float(work[n - 1, n - 1])

    # solve m X = I  (L then U), using stored L in lower part
    rhs = np.eye(n, dtype=np.float32)
    # apply permutations
    for j, p in enumerate(piv):
        if p != j:
            t = rhs[j].copy()
            rhs[j] = rhs[p]
            rhs[p] = t
    for j in range(n):
        if j + 1 < n:
            l = work[j + 1:, j]
            rhs[j + 1:, :] = (rhs[j + 1:, :] - l[:, None] * rhs[j][None, :]).astype(np.float32)
    for j in range(n - 1, -1, -1):
        d = work[j, j]
        rhs[j, :] = (rhs[j, :] / d).astype(np.float32)
        if j > 0:
            u = work[:j, j]
            rhs[:j, :] = (rhs[:j, :] - u[:, None] * rhs[j][None, :]).astype(np.float32)
    inv = rhs if trans_first else rhs
    if trans_first:
        inv = rhs.T.astype(np.float32)
    return inv, info, piv, last_diag


def summarize(t):
    f = t.reshape(-1)
    fin = f[torch.isfinite(f)]
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "elements": int(f.numel()),
        "nan_count": int(torch.isnan(f).sum().item()),
        "posinf_count": int(torch.isinf(f).sum().item() and int((f == float("inf")).sum().item())),
        "neginf_count": int((f == -float("inf")).sum().item()),
        "finite_min": float(fin.min().item()) if fin.numel() else None,
        "finite_max": float(fin.max().item()) if fin.numel() else None,
        "values": t.tolist() if f.numel() <= 16 else None,
    }


out = {
    "torch": torch.__version__,
    "numpy": np.__version__,
    "torch_npu": getattr(torch, "npu", None) and __import__("torch_npu").__version__,
    "cases": {},
}

cases = [
    ("c1_sing3", {"dtype": "float32", "shape": [3, 3], "singular": True}, 1730),
    ("c0_reg3", {"dtype": "float32", "shape": [3, 3], "singular": False}, 1729),
    ("c3_sing7", {"dtype": "float32", "shape": [2, 7, 7], "singular": True}, 1732),
    ("c2_reg7", {"dtype": "float32", "shape": [2, 7, 7], "singular": False}, 1731),
]

for name, spec, seed in cases:
    A_npu = gen_inputs(spec, seed, "npu")
    A_cpu = A_npu.detach().cpu()
    rows_equal = bool(torch.equal(A_cpu[..., 0, :], A_cpu[..., -1, :])) if spec.get("singular") else None
    rec = {
        "spec": spec,
        "seed": seed,
        "rows0_last_bitwise_equal": rows_equal,
        "A": A_cpu.reshape(-1, spec["shape"][-1]).tolist() if spec["shape"][-1] <= 7 else None,
    }
    ref_npu = torch.linalg.inv_ex(A_npu, check_errors=False)
    ref_cpu = torch.linalg.inv_ex(A_cpu, check_errors=False)
    rec["npu"] = {"inverse": summarize(ref_npu.inverse.cpu()), "info": ref_npu.info.cpu().tolist(),
                  "info_dtype": str(ref_npu.info.dtype), "type": type(ref_npu).__name__}
    rec["cpu"] = {"inverse": summarize(ref_cpu.inverse), "info": ref_cpu.info.tolist()}
    rec["npu_vs_cpu_bitwise"] = bool(torch.equal(ref_npu.inverse.cpu(), ref_cpu.inverse))

    # scalar LU variants on the CPU copy of the exact frozen input
    a0 = A_cpu.reshape(-1, *spec["shape"][-2:])
    for vname, tflag in (("lu_direct", False), ("lu_transposed_image", True)):
        res = []
        for b in range(a0.shape[0]):
            inv, info, piv, lastd = lu_inv(a0[b].numpy(), trans_first=tflag)
            res.append({"info": info, "last_diag": lastd, "pivots": piv,
                        "inv": inv.tolist()})
        rec[vname] = res
    out["cases"][name] = rec

with open(os.path.join(ART, "probe_ref.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps({k: {"npu_info": v["npu"]["info"], "cpu_info": v["cpu"]["info"],
                      "npu_vs_cpu_bitwise": v["npu_vs_cpu_bitwise"],
                      "rows_equal": v["rows0_last_bitwise_equal"],
                      "lu_direct_info": [r["info"] for r in v["lu_direct"]],
                      "lu_trans_info": [r["info"] for r in v["lu_transposed_image"]],
                      "lu_direct_lastdiag": [r["last_diag"] for r in v["lu_direct"]],
                      "lu_trans_lastdiag": [r["last_diag"] for r in v["lu_transposed_image"]]}
                  for k, v in out["cases"].items()}, indent=1))
print("torch", out["torch"], "torch_npu", out["torch_npu"])
