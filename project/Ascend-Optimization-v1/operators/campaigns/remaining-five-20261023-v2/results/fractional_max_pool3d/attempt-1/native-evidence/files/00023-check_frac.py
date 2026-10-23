"""Device-side numeric gate for the fractional_max_pool3d candidate.

Compares tmp/main.py::run against the PyTorch reference (F.fractional_max_pool3d)
on the frozen official workloads and on supplementary layouts/special values.
Writes the JSON artifact after every case so a timeout still leaves evidence.
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
OUT = os.path.join(ART, "frac_check.json")
STATE = {"cases": [], "timings": []}


def flush():
    fail = [c["name"] for c in STATE["cases"] if c.get("status") != "pass"]
    with open(OUT, "w") as fh:
        json.dump({"cases": STATE["cases"], "timings": STATE["timings"],
                   "failures": fail,
                   "n_pass": len(STATE["cases"]) - len(fail),
                   "n_cases": len(STATE["cases"])}, fh, indent=2)


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


def run_case(name, x, ks, osz, samples, return_indices=True, ratio=None, tol=None):
    entry = {"name": name, "shape": list(x.shape), "dtype": str(x.dtype),
             "kernel_size": list(ks) if not isinstance(ks, int) else ks,
             "output_size": list(osz) if osz is not None else None,
             "output_ratio": ratio, "return_indices": return_indices,
             "stride": list(x.stride())}
    x_orig = x.clone()
    try:
        ref = F.fractional_max_pool3d(
            x, ks, output_size=osz, output_ratio=ratio,
            return_indices=return_indices, _random_samples=samples)
    except Exception as exc:  # noqa: BLE001
        entry["status"] = "reference_error"
        entry["error"] = f"{type(exc).__name__}: {exc}"
        STATE["cases"].append(entry)
        flush()
        return entry
    try:
        got = cand.run(
            x, ks, output_size=osz, output_ratio=ratio,
            return_indices=return_indices, _random_samples=samples)
    except Exception as exc:  # noqa: BLE001
        entry["status"] = "candidate_error"
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["trace"] = traceback.format_exc()[-1200:]
        STATE["cases"].append(entry)
        flush()
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
    entry["status"] = "pass" if (ok and entry["out_dtype_ok"] and
                                 entry["out_shape_ok"] and
                                 entry.get("idx_ratio", 1.0) == 1.0) else "fail"
    STATE["cases"].append(entry)
    flush()
    return entry


def time_case(name, x, ks, osz, samples, iters=20):
    entry = {"name": name, "iters": iters}
    try:
        for _ in range(3):
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
        entry["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        entry["status"] = "error"
        entry["error"] = f"{type(exc).__name__}: {exc}"
    STATE["timings"].append(entry)
    flush()
    return entry


OFFICIAL = [
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


def main():
    flush()
    for name, shape, dtype, seed, ks, osz in OFFICIAL:
        x, s = gen(shape, dtype, seed)
        run_case("official-" + name, x, ks, osz, s)

    # ratio path special values (exact 2x tiling shape)
    for dtype in (torch.float16, torch.float32):
        g = torch.Generator().manual_seed(120)
        s = torch.rand((1, 1, 3), generator=g).to(dtype=dtype, device=DEV)
        base = torch.randn(1, 1, 8, 8, 8, generator=torch.Generator().manual_seed(121)).to(
            dtype=dtype, device=DEV)
        nan = base.clone()
        flat = nan.view(-1)
        flat[3] = float("nan")
        flat[40] = float("nan")
        run_case(f"sup-nan-mixed-{str(dtype)[6:]}", nan, [2, 2, 2], [4, 4, 4], s)
        allnan = torch.full_like(base, float("nan"))
        run_case(f"sup-nan-all-{str(dtype)[6:]}", allnan, [2, 2, 2], [4, 4, 4], s)
        run_case(f"sup-neg-inf-{str(dtype)[6:]}", torch.full_like(base, float("-inf")),
                 [2, 2, 2], [4, 4, 4], s)
        run_case(f"sup-ties-{str(dtype)[6:]}", torch.ones_like(base),
                 [2, 2, 2], [4, 4, 4], s)
        pinf = base.clone()
        pinf.view(-1)[5] = float("inf")
        run_case(f"sup-pos-inf-{str(dtype)[6:]}", pinf, [2, 2, 2], [4, 4, 4], s)

    dtype = torch.float16
    # integer ratio with a non power of two window (padded lanes)
    x9, s9 = gen([2, 2, 9, 9, 9], dtype, 110)
    run_case("sup-ratio-k3", x9, [3, 3, 3], [3, 3, 3], s9)
    # output extent that does not divide the block
    x7, s7 = gen([1, 3, 14, 14, 14], dtype, 111)
    run_case("sup-ratio-7x7", x7, [2, 2, 2], [7, 7, 7], s7)
    # 4D unbatched input, samples (1, C, 3) as the ATen op requires
    x4 = torch.randn(3, 8, 8, 8, generator=torch.Generator().manual_seed(105)).to(
        dtype=dtype, device=DEV)
    s4 = torch.rand(1, 3, 3, generator=torch.Generator().manual_seed(106)).to(
        dtype=dtype, device=DEV)
    run_case("sup-4d-input", x4, [2, 2, 2], [4, 4, 4], s4)
    # non-contiguous and stride-2 views
    x, s = gen([2, 2, 8, 10, 12], dtype, 107)
    run_case("sup-transposed", x.transpose(-1, -2), [2, 2, 2], [4, 5, 6], s)
    xb = torch.randn((2, 2, 8, 8, 16), generator=torch.Generator().manual_seed(108)).to(
        dtype=dtype, device=DEV)
    sxb = torch.rand(2, 2, 3, generator=torch.Generator().manual_seed(109)).to(
        dtype=dtype, device=DEV)
    run_case("sup-stride2-w", xb[..., ::2], [2, 2, 2], [4, 4, 4], sxb)
    # general path specials
    xo, so = gen([2, 3, 9, 11, 13], dtype, 101)
    run_case("sup-oddratio-k3", xo, [3, 3, 3], [4, 4, 4], so)
    xu, su = gen([2, 3, 8, 8, 8], dtype, 102)
    run_case("sup-unit-out", xu, [2, 2, 2], [1, 1, 1], su)
    run_case("sup-unit-out-d", xu, [2, 2, 2], [1, 4, 4], su)
    xn, sn = gen([1, 4, 4, 4, 4], dtype, 103)
    run_case("sup-no-indices", xn, [2, 2, 2], [2, 2, 2], sn, return_indices=False)
    xr, sr = gen([3, 5, 12, 10, 14], dtype, 104)
    run_case("sup-ratio-half", xr, [2, 2, 2], None, sr, ratio=0.5)

    for name, shape, dtype_, seed, ks, osz in OFFICIAL[3:]:
        x, s = gen(shape, dtype_, seed)
        time_case("timing-" + name, x, ks, osz, s)

    flush()
    print(json.dumps({"failures": [c["name"] for c in STATE["cases"]
                                   if c.get("status") != "pass"],
                      "n_cases": len(STATE["cases"]),
                      "timings": STATE["timings"]}))


main()
