"""Validate round-2 candidate on target: exact compare() semantics for all 18
correctness workloads, direct-launch engagement, and timing vs the reference.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main as cand  # noqa: E402

ART = Path(os.environ.get("KGS_DEBUG_ARTIFACTS", ".")) / "probe3.json"
DEV = "npu:0"
EDGE = [0., -0., float("nan"), float("inf"), -float("inf"), -1., 1.]
out = {"cases": {}, "timing": {}, "direct_state": {}}


def gen(spec, seed):
    dtype = getattr(torch, spec["dtype"])
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(spec["shape"], generator=g).to(device=DEV, dtype=dtype)
    if spec.get("special") and x.numel() >= 7:
        x.flatten()[:7] = torch.tensor(EDGE, dtype=dtype, device=DEV)
    if spec.get("strided"):
        base = torch.empty((*x.shape, 2), dtype=dtype, device=DEV)
        base[..., 0] = x
        x = base[..., 0]
    return x


def check(ref, got, rtol, atol):
    """Replicates kernelgen_server.evaluation.compare.compare for one tensor."""
    ref = ref.detach().to("cpu")
    got = got.detach().to("cpu")
    if ref.shape != got.shape:
        return {"ok": False, "why": "shape", "ref": list(ref.shape), "got": list(got.shape)}
    if ref.dtype != got.dtype:
        return {"ok": False, "why": "dtype"}
    r = ref.to(torch.float32)
    c = got.to(torch.float32)
    both_finite = torch.isfinite(r) & torch.isfinite(c)
    same_inf = torch.isinf(r) & torch.isinf(c) & torch.eq(r, c)
    same_nan = torch.isnan(r) & torch.isnan(c)
    if not (both_finite | same_inf | same_nan).all().item():
        bad = torch.nonzero(~(both_finite | same_inf | same_nan)).flatten()[:8].tolist()
        return {"ok": False, "why": "non-finite", "idx": bad,
                "ref_at": [float(r.flatten()[i]) for i in bad],
                "got_at": [float(c.flatten()[i]) for i in bad]}
    abs_err = torch.where(both_finite, (c - r).abs(), torch.zeros_like(r))
    den = torch.where(both_finite, r.abs(), torch.zeros_like(r))
    ok = same_inf | same_nan | (both_finite & (abs_err <= atol + rtol * den))
    ratio = float(ok.sum().item()) / max(1, ref.numel())
    # bit-level agreement diagnostic (sign of zero included)
    rb = ref.view(torch.int16 if ref.element_size() == 2 else torch.int32)
    gb = got.view(torch.int16 if got.element_size() == 2 else torch.int32)
    nbit = int((rb != gb).sum().item())
    return {"ok": bool(ratio >= 1.0), "matched_ratio": ratio,
            "max_abs": float(abs_err.max().item()), "bit_diff": nbit}


CORRECTNESS = [
    ("c0", {"dtype": "float16", "shape": [1], "strided": False, "special": True}, 1729, 1.5e-2),
    ("c1", {"dtype": "float16", "shape": [1], "strided": True, "special": True}, 1730, 1.5e-2),
    ("c2", {"dtype": "float16", "shape": [1025], "strided": False, "special": True}, 1731, 1.5e-2),
    ("c3", {"dtype": "float16", "shape": [1025], "strided": True, "special": True}, 1732, 1.5e-2),
    ("c4", {"dtype": "float16", "shape": [7, 9], "strided": False, "special": True}, 1733, 1.5e-2),
    ("c5", {"dtype": "float16", "shape": [7, 9], "strided": True, "special": True}, 1734, 1.5e-2),
    ("c6", {"dtype": "bfloat16", "shape": [1], "strided": False, "special": True}, 1735, 1.5e-2),
    ("c7", {"dtype": "bfloat16", "shape": [1], "strided": True, "special": True}, 1736, 1.5e-2),
    ("c8", {"dtype": "bfloat16", "shape": [1025], "strided": False, "special": True}, 1737, 1.5e-2),
    ("c9", {"dtype": "bfloat16", "shape": [1025], "strided": True, "special": True}, 1738, 1.5e-2),
    ("c10", {"dtype": "bfloat16", "shape": [7, 9], "strided": False, "special": True}, 1739, 1.5e-2),
    ("c11", {"dtype": "bfloat16", "shape": [7, 9], "strided": True, "special": True}, 1740, 1.5e-2),
    ("c12", {"dtype": "float32", "shape": [1], "strided": False, "special": True}, 1741, 1e-4),
    ("c13", {"dtype": "float32", "shape": [1], "strided": True, "special": True}, 1742, 1e-4),
    ("c14", {"dtype": "float32", "shape": [1025], "strided": False, "special": True}, 1743, 1e-4),
    ("c15", {"dtype": "float32", "shape": [1025], "strided": True, "special": True}, 1744, 1e-4),
    ("c16", {"dtype": "float32", "shape": [7, 9], "strided": False, "special": True}, 1745, 1e-4),
    ("c17", {"dtype": "float32", "shape": [7, 9], "strided": True, "special": True}, 1746, 1e-4),
]

TIMING = [
    ("t0", "float16", [1024], 1729), ("t1", "float16", [65536], 1730),
    ("t2", "float16", [1048576], 1731), ("t3", "float16", [4096, 4096], 1732),
    ("t4", "bfloat16", [1024], 1733), ("t5", "bfloat16", [65536], 1734),
    ("t6", "bfloat16", [1048576], 1735), ("t7", "bfloat16", [4096, 4096], 1736),
    ("t8", "float32", [1024], 1737), ("t9", "float32", [65536], 1738),
    ("t10", "float32", [1048576], 1739), ("t11", "float32", [4096, 4096], 1740),
]


def bench(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    best = []
    for _ in range(3):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        best.append((time.perf_counter() - t0) / iters * 1e3)
    best.sort()
    return best[1]


def main():
    for name, spec, seed, tol in CORRECTNESS:
        try:
            a = gen(spec, seed)
            ref = a.clone().neg_()
            got = cand.run(a)
            torch.npu.synchronize()
            rec = check(ref, got, tol, tol)
            rec["alias_ok"] = got.data_ptr() == a.data_ptr()
            rec["direct"] = cand._DIRECT[0]
            out["cases"][name] = rec
        except Exception:
            out["cases"][name] = {"ok": False, "error": traceback.format_exc()[-700:]}

    out["direct_state"] = {"direct": cand._DIRECT[0],
                           "entries": len(cand._ENTRIES),
                           "keys": [str(k) for k in cand._ENTRIES]}

    for name, dtype, shape, seed in TIMING:
        a = gen({"dtype": dtype, "shape": shape}, seed)
        b = a.clone()
        n = a.numel()
        iters = 200 if n <= 65536 else 60
        try:
            t_cand = bench(lambda: cand.run(a), iters)
            t_ref = bench(lambda: b.neg_(), iters)
            out["timing"][name] = {"n": n, "dtype": dtype,
                                   "cand_ms": round(t_cand, 5),
                                   "ref_ms": round(t_ref, 5),
                                   "speedup": round(t_ref / t_cand, 4)}
        except Exception:
            out["timing"][name] = {"error": traceback.format_exc()[-500:]}
        del a, b
    out["direct_state_after"] = {"direct": cand._DIRECT[0],
                                 "entries": len(cand._ENTRIES)}


try:
    main()
except Exception:
    out["fatal"] = traceback.format_exc()
finally:
    ART.parent.mkdir(parents=True, exist_ok=True)
    ART.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str)[:9000])
