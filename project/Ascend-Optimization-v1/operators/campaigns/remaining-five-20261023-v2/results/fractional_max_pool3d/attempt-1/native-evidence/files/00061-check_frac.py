"""Device-side numeric gate for the fractional_max_pool3d candidate.

Compares tmp/main.py::run against the PyTorch reference (F.fractional_max_pool3d)
on the frozen official workloads and on supplementary layouts/special values.
Diagnostic evidence only; the authoritative result is eval_round.
"""
import importlib
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.dirname(HERE)
sys.path.insert(0, TMP)

import torch
import torch.nn.functional as F

cand = importlib.import_module("main")

DEV = "npu"
ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")

TOL = {
    torch.float16: (1.5e-2, 1.5e-2),
    torch.bfloat16: (1.5e-2, 1.5e-2),
    torch.float32: (1e-4, 1e-4),
}


def gen(shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g).to(dtype=dtype, device=DEV)
    samples = torch.rand((shape[0], shape[1], 3), generator=g).to(
        dtype=dtype, device=DEV
    )
    return x, samples


def nanaware_equal(a, b):
    an, bn = torch.isnan(a), torch.isnan(b)
    if not torch.equal(an, bn):
        return False
    return torch.equal(torch.where(an, torch.zeros_like(a), a),
                       torch.where(bn, torch.zeros_like(b), b))


def run_case(rec, name, x, ks, osz, samples, return_indices=True,
             ratio=None, tol=None):
    entry = {"name": name, "shape": list(x.shape), "dtype": str(x.dtype),
             "kernel_size": list(ks) if not isinstance(ks, int) else ks,
             "output_size": list(osz) if osz is not None else None,
             "output_ratio": ratio, "return_indices": return_indices,
             "strided": [list(s) for s in [x.stride()]]}
    x_orig = x.clone()
    try:
        ref = F.fractional_max_pool3d(
            x, ks, output_size=osz, output_ratio=ratio,
            return_indices=return_indices, _random_samples=samples)
    except Exception as exc:  # noqa: BLE001
        entry["status"] = "reference_error"
        entry["error"] = f"{type(exc).__name__}: {exc}"
        rec.append(entry)
        return entry
    try:
        got = cand.run(
            x, ks, output_size=osz, output_ratio=ratio,
            return_indices=return_indices, _random_samples=samples)
    except Exception as exc:  # noqa: BLE001
        entry["status"] = "candidate_error"
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["trace"] = traceback.format_exc()[-1500:]
        rec.append(entry)
        return entry

    entry["input_untouched"] = bool(torch.equal(x_orig, x))
    if return_indices:
        ref_v, ref_i = ref
        got_v, got_i = got
    else:
        ref_v, got_v = ref, got
        ref_i = got_i = None
    entry["out_dtype_ok"] = str(ref_v.dtype) == str(got_v.dtype)
    entry["out_shape_ok"] = list(ref_v.shape) == list(got_v.shape)
    if ref_i is not None:
        entry["idx_dtype_ok"] = str(ref_i.dtype) == str(got_i.dtype)
        idx_match = (ref_i == got_i)
        entry["idx_ratio"] = float(idx_match.float().mean().item())
        entry["idx_mismatch"] = int((~idx_match).sum().item())
    atol, rtol = tol if tol is not None else TOL[x.dtype]
    a = got_v.float()
    b = ref_v.float()
    diff = (a - b).abs()
    entry["max_abs"] = float(torch.nan_to_num(diff, nan=float("inf")).max().item()) \
        if diff.numel() else 0.0
    ok = nanaware_equal(a, b) if not torch.isfinite(b).all() else bool(
        torch.all(diff <= atol + rtol * b.abs()))
    entry["values_ok"] = ok
    entry["atol"] = atol
    entry["rtol"] = rtol
    entry["status"] = "pass" if (ok and entry["out_dtype_ok"] and
                                 entry["out_shape_ok"] and
                                 entry.get("idx_ratio", 1.0) == 1.0) else "fail"
    rec.append(entry)
    return entry


def time_case(rec, name, x, ks, osz, samples, iters=50):
    entry = {"name": name, "iters": iters}
    try:
        for _ in range(5):
            cand.run(x, ks, output_size=osz, return_indices=True,
                     _random_samples=samples)
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            cand.run(x, ks, output_size=osz, return_indices=True,
                     _random_samples=samples)
        end.record()
        torch.npu.synchronize()
        entry["candidate_ms"] = start.elapsed_time(end) / iters

        for _ in range(3):
            F.fractional_max_pool3d(x, ks, output_size=osz, return_indices=True,
                                    _random_samples=samples)
        torch.npu.synchronize()
        start.record()
        for _ in range(iters):
            F.fractional_max_pool3d(x, ks, output_size=osz, return_indices=True,
                                    _random_samples=samples)
        end.record()
        torch.npu.synchronize()
        entry["reference_ms"] = start.elapsed_time(end) / iters
        entry["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        entry["status"] = "error"
        entry["error"] = f"{type(exc).__name__}: {exc}"
    rec.append(entry)
    return entry


def main():
    rec = []
    official = [
        ("correctness-0", [1, 2, 7, 9, 11], torch.float16, 1729, [2, 2, 2], [3, 4, 5]),
        ("correctness-1", [1, 2, 7, 9, 11], torch.bfloat16, 1730, [2, 2, 2], [3, 4, 5]),
        ("correctness-2", [1, 2, 7, 9, 11], torch.float32, 1731, [2, 2, 2], [3, 4, 5]),
        ("timing-0", [1, 8, 16, 16, 16], torch.float16, 1729, [2, 2, 2], [8, 8, 8]),
        ("timing-1", [2, 16, 32, 32, 32], torch.float16, 1730, [2, 2, 2], [16, 16, 16]),
        ("timing-2", [1, 8, 16, 16, 16], torch.bfloat16, 1731, [2, 2, 2], [8, 8, 8]),
        ("timing-3", [2, 16, 32, 32, 32], torch.bfloat16, 1732, [2, 2, 2], [16, 16, 16]),
        ("timing-4", [1, 8, 16, 16, 16], torch.float32, 1733, [2, 2, 2], [8, 8, 8]),
        ("timing-5", [2, 16, 32, 32, 32], torch.float32, 1734, [2, 2, 2], [16, 16, 16]),
    ]
    for name, shape, dtype, seed, ks, osz in official:
        x, s = gen(shape, dtype, seed)
        run_case(rec, "official-" + name, x, ks, osz, s)

    # Supplementary semantics: ratios, odd shapes, 4D, strided, special values.
    sup = []
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        x, s = gen([2, 3, 9, 11, 13], dtype, 101)
        sup.append(("oddratio-k3-o4", x, [3, 3, 3], [4, 4, 4], s, True))
        x, s = gen([2, 3, 8, 8, 8], dtype, 102)
        sup.append(("unit-out", x, [2, 2, 2], [1, 1, 1], s, True))
        sup.append(("unit-out-d", x, [2, 2, 2], [1, 4, 4], s, True))
        x, s = gen([1, 4, 4, 4, 4], dtype, 103)
        sup.append(("no-indices", x, [2, 2, 2], [2, 2, 2], s, False))
        x, s = gen([3, 5, 12, 10, 14], dtype, 104)
        sup.append(("ratio-half", x, [2, 2, 2], None, s, True))
        # 4D unbatched input, samples (C, 3)
        x4 = torch.randn(3, 8, 8, 8, generator=torch.Generator().manual_seed(105)).to(
            dtype=dtype, device=DEV)
        s4 = torch.rand(3, 3, generator=torch.Generator().manual_seed(106)).to(
            dtype=dtype, device=DEV)
        sup.append(("4d-input", x4, [2, 2, 2], [4, 4, 4], s4, True))
        # non-contiguous (transposed last two dims) and stride-2 view
        x, s = gen([2, 2, 8, 10, 12], dtype, 107)
        xt = x.transpose(-1, -2)
        sup.append(("transposed", xt, [2, 2, 2], [4, 5, 6], s, True))
        xb = torch.empty((2, 2, 8, 8, 16), dtype=dtype, device=DEV)
        xb[..., ::2] = x[..., :16] if x.shape[-1] >= 16 else xb[..., ::2]
        xb[..., ::2] = torch.randn((2, 2, 8, 8, 8), generator=torch.Generator().manual_seed(108)).to(dtype=dtype, device=DEV)
        sb, _ = gen([2, 2, 8, 8, 8], dtype, 109)
        sup.append(("stride2-w", xb[..., ::2], [2, 2, 2], [4, 4, 4], sb, True))

    # Special values on a shape that takes the integer-ratio path.
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        g = torch.Generator().manual_seed(120)
        s = torch.rand((1, 1, 3), generator=g).to(dtype=dtype, device=DEV)
        base = torch.randn(1, 1, 8, 8, 8, generator=torch.Generator().manual_seed(121)).to(
            dtype=dtype, device=DEV)
        nan = base.clone()
        nan.view(-1)[3] = float("nan")
        nan.view(-1)[40] = float("nan")
        sup.append(("nan-mixed", nan, [2, 2, 2], [4, 4, 4], s, True))
        allnan = torch.full_like(base, float("nan"))
        sup.append(("nan-all", allnan, [2, 2, 2], [4, 4, 4], s, True))
        ninf = torch.full_like(base, float("-inf"))
        sup.append(("neg-inf", ninf, [2, 2, 2], [4, 4, 4], s, True))
        ones = torch.ones_like(base)
        sup.append(("ties", ones, [2, 2, 2], [4, 4, 4], s, True))
        pinf = base.clone()
        pinf.view(-1)[5] = float("inf")
        sup.append(("pos-inf", pinf, [2, 2, 2], [4, 4, 4], s, True))

    for name, x, ks, osz, s, ri in sup:
        ratio = 0.5 if name == "ratio-half" else None
        run_case(rec, "supp-" + name, x, ks, osz, s, return_indices=ri, ratio=ratio)

    timings = []
    for name, shape, dtype, seed, ks, osz in official[3:]:
        x, s = gen(shape, dtype, seed)
        time_case(timings, "timing-" + name, x, ks, osz, s)

    summary = {
        "cases": rec,
        "timings": timings,
        "failures": [c["name"] for c in rec if c.get("status") != "pass"],
        "n_pass": sum(1 for c in rec if c.get("status") == "pass"),
        "n_cases": len(rec),
    }
    with open(os.path.join(ART, "frac_check.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({"failures": summary["failures"],
                      "n_pass": summary["n_pass"],
                      "n_cases": summary["n_cases"]}))
    for c in rec:
        if c.get("status") != "pass":
            print("FAIL", json.dumps(c)[:1200])
    for t in timings:
        print("TIME", json.dumps(t))


main()
