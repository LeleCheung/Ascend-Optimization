"""Round-4 validation: exact compare() semantics on all 30 frozen workload
shapes and timing, with the trimmed per-call host path.
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

ART = Path(os.environ.get("KGS_DEBUG_ARTIFACTS", ".")) / "probe7.json"
DEV = "npu:0"
EDGE = [0., -0., float("nan"), float("inf"), -float("inf"), -1., 1.]
out = {"cases": {}, "state": {}, "raw": {}}


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
    ref = ref.detach().to("cpu")
    got = got.detach().to("cpu")
    if ref.shape != got.shape or ref.dtype != got.dtype:
        return {"ok": False, "why": "shape/dtype"}
    r = ref.to(torch.float32)
    c = got.to(torch.float32)
    both_finite = torch.isfinite(r) & torch.isfinite(c)
    same_inf = torch.isinf(r) & torch.isinf(c) & torch.eq(r, c)
    same_nan = torch.isnan(r) & torch.isnan(c)
    if not (both_finite | same_inf | same_nan).all().item():
        bad = torch.nonzero(~(both_finite | same_inf | same_nan)).flatten()[:8].tolist()
        return {"ok": False, "why": "non-finite", "idx": bad}
    abs_err = torch.where(both_finite, (c - r).abs(), torch.zeros_like(r))
    den = torch.where(both_finite, r.abs(), torch.zeros_like(r))
    ok = same_inf | same_nan | (both_finite & (abs_err <= atol + rtol * den))
    return {"ok": bool(ok.all().item()), "max_abs": float(abs_err.max().item())}


SPECS = [
    ("c0", "float16", [1], False, True, 1729, 1.5e-2),
    ("c1", "float16", [1], True, True, 1730, 1.5e-2),
    ("c2", "float16", [1025], False, True, 1731, 1.5e-2),
    ("c3", "float16", [1025], True, True, 1732, 1.5e-2),
    ("c4", "float16", [7, 9], False, True, 1733, 1.5e-2),
    ("c5", "float16", [7, 9], True, True, 1734, 1.5e-2),
    ("c6", "bfloat16", [1], False, True, 1735, 1.5e-2),
    ("c7", "bfloat16", [1], True, True, 1736, 1.5e-2),
    ("c8", "bfloat16", [1025], False, True, 1737, 1.5e-2),
    ("c9", "bfloat16", [1025], True, True, 1738, 1.5e-2),
    ("c10", "bfloat16", [7, 9], False, True, 1739, 1.5e-2),
    ("c11", "bfloat16", [7, 9], True, True, 1740, 1.5e-2),
    ("c12", "float32", [1], False, True, 1741, 1e-4),
    ("c13", "float32", [1], True, True, 1742, 1e-4),
    ("c14", "float32", [1025], False, True, 1743, 1e-4),
    ("c15", "float32", [1025], True, True, 1744, 1e-4),
    ("c16", "float32", [7, 9], False, True, 1745, 1e-4),
    ("c17", "float32", [7, 9], True, True, 1746, 1e-4),
    ("t0", "float16", [1024], False, False, 1729, 1.5e-2),
    ("t1", "float16", [65536], False, False, 1730, 1.5e-2),
    ("t2", "float16", [1048576], False, False, 1731, 1.5e-2),
    ("t3", "float16", [4096, 4096], False, False, 1732, 1.5e-2),
    ("t4", "bfloat16", [1024], False, False, 1733, 1.5e-2),
    ("t5", "bfloat16", [65536], False, False, 1734, 1.5e-2),
    ("t6", "bfloat16", [1048576], False, False, 1735, 1.5e-2),
    ("t7", "bfloat16", [4096, 4096], False, False, 1736, 1.5e-2),
    ("t8", "float32", [1024], False, False, 1737, 1e-4),
    ("t9", "float32", [65536], False, False, 1738, 1e-4),
    ("t10", "float32", [1048576], False, False, 1739, 1e-4),
    ("t11", "float32", [4096, 4096], False, False, 1740, 1e-4),
]


def bench(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(3):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) / iters * 1e3)
    ts.sort()
    return ts[1]


def main():
    for name, dtype, shape, strided, special, seed, tol in SPECS:
        try:
            spec = {"dtype": dtype, "shape": shape, "strided": strided, "special": special}
            a = gen(spec, seed)
            ref = a.clone().neg_()
            got = cand.run(a)
            torch.npu.synchronize()
            rec = check(ref, got, tol, tol)
            rec["alias"] = got.data_ptr() == a.data_ptr()
            if name.startswith("t"):
                b = gen(spec, seed)
                n = a.numel()
                iters = 300 if n <= 65536 else 60
                tc = bench(lambda: cand.run(a), iters)
                tr = bench(lambda: b.neg_(), iters)
                rec.update({"cand_ms": round(tc, 5), "ref_ms": round(tr, 5),
                            "speedup": round(tr / tc, 4)})
                # a second, independent measurement of the same call
                rec["cand_ms_2"] = round(bench(lambda: cand.run(a), iters), 5)
                del b
            out["cases"][name] = rec
            del a, ref, got
        except Exception:
            out["cases"][name] = {"ok": False, "error": traceback.format_exc()[-600:]}
    out["state"] = {"plans": len(cand._PLANS), "stream_reader": str(cand._STREAM),
                    "bad_plans": [str(k) for k, v in cand._PLANS.items() if v is False]}


try:
    main()
except Exception:
    out["fatal"] = traceback.format_exc()
finally:
    ART.parent.mkdir(parents=True, exist_ok=True)
    ART.write_text(json.dumps(out, indent=2, default=str))
    failed = [k for k, v in out["cases"].items() if not v.get("ok")]
    print(json.dumps({"failed": failed, "state": out.get("state"),
                      **{k: {kk: vv for kk, vv in v.items() if kk in
                             ("ok", "cand_ms", "cand_ms_2", "ref_ms", "speedup")}
                         for k, v in out["cases"].items() if k.startswith("t")}},
                     indent=2, default=str)[:5000])
