"""Capture the aten-level decomposition of torch 2.9 CPU linalg_inv_ex.

If the op is a composite, a TorchDispatchMode sees its sub-ops, which tells us
exactly which leaf routines to emulate.  Also reports solve_ex's info and the
LU-free alternatives.
"""

import json
import os

import torch
from torch.utils._python_dispatch import TorchDispatchMode

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


class Tracer(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        shapes = []
        for a in args:
            if isinstance(a, torch.Tensor):
                shapes.append((list(a.shape), str(a.dtype), list(a.stride())))
        self.calls.append(
            {"op": str(func), "args": shapes,
             "kw": {k: str(v)[:40] for k, v in kwargs.items()}}
        )
        return func(*args, **kwargs)


def trace(fn):
    t = Tracer()
    try:
        with t:
            out = fn()
        return t.calls, None
    except Exception as exc:
        return t.calls, repr(exc)[:300]


A = build({"shape": [3, 3], "singular": True}, 1730)
A3ns = build({"shape": [3, 3], "singular": False}, 1729)
I3 = torch.eye(3, dtype=torch.float32)

report = {}
for label, fn in {
    "inv_ex_singular": lambda: torch.linalg.inv_ex(A, check_errors=False),
    "inv_singular": lambda: torch.linalg.inv(A),
    "solve_ex_singular": lambda: torch.linalg.solve_ex(A, I3),
    "solve_singular": lambda: torch.linalg.solve(A, I3),
    "inv_ex_nonsingular": lambda: torch.linalg.inv_ex(A3ns, check_errors=False),
}.items():
    calls, err = trace(fn)
    report[label] = {"n_ops": len(calls), "ops": calls, "error": err}

# solve_ex info on the singular input
se = torch.linalg.solve_ex(A, I3)
report["solve_ex_info"] = se[1].tolist()
report["solve_ex_result"] = se[0].tolist()
report["inv_ex_batched_info"] = torch.linalg.solve_ex(
    build({"shape": [2, 7, 7], "singular": True}, 1732),
    torch.eye(7).expand(2, 7, 7).contiguous(),
)[1].tolist()

# alternative decompositions that might reproduce the reference
ref = torch.linalg.inv_ex(A, check_errors=False)[0]
alts = {}
for name, thunk in {
    "cholesky_based": lambda: torch.cholesky_inverse(torch.linalg.cholesky(A @ A.T + 1e-3 * torch.eye(3))),
    "lstsq": lambda: torch.linalg.lstsq(A, I3).solution,
    "pinv_rcond_none": lambda: torch.linalg.pinv(A, rtol=None, atol=None),
}.items():
    try:
        v = thunk()
        alts[name] = bool(torch.equal(v, ref)) if v.shape == ref.shape else "shape"
    except Exception as exc:
        alts[name] = repr(exc)[:120]
report["alternatives"] = alts

with open(os.path.join(OUT, "probe3.json"), "w") as f:
    json.dump(report, f, indent=1)
print(json.dumps({k: report[k] for k in ("inv_ex_singular", "solve_ex_singular")}, indent=1)[:6000])
print("solve_ex_info", report["solve_ex_info"], "batched", report["inv_ex_batched_info"])
