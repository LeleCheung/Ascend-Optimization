"""Bounded target-side probe for the log2_ candidate.

1. Rebuilds the exact frozen workload inputs (same gen_inputs logic) and checks
   the candidate `run(A)` against the torch oracle ``A.log2_()``: value
   agreement inside the frozen tolerance, output dtype, in-place identity
   (same object / same data_ptr / unchanged strides) and, for strided views,
   that the parent storage padding slots were not touched.
2. Measures the per-call walltime of the candidate vs the torch oracle for a few
   representative sizes, plus the cost of the individual host-side pieces
   (data_ptr, raw current-stream query, torch.npu.current_stream()).

Writes JSON under $KGS_DEBUG_ARTIFACTS (or ./).
"""

import importlib.util
import json
import os
import sys
import time
import traceback

import torch

DEV = "npu:0"
torch.npu.set_device(0)

ART = os.environ.get("KGS_DEBUG_ARTIFACTS") or "."
if ART and not os.path.isdir(ART):
    os.makedirs(ART, exist_ok=True)

CORR = []
for i, (dt, shape, strided, seed) in enumerate([
        ("float16", [1], False, 1729), ("float16", [1], True, 1730),
        ("float16", [1025], False, 1731), ("float16", [1025], True, 1732),
        ("float16", [7, 9], False, 1733), ("float16", [7, 9], True, 1734),
        ("bfloat16", [1], False, 1735), ("bfloat16", [1], True, 1736),
        ("bfloat16", [1025], False, 1737), ("bfloat16", [1025], True, 1738),
        ("bfloat16", [7, 9], False, 1739), ("bfloat16", [7, 9], True, 1740),
        ("float32", [1], False, 1741), ("float32", [1], True, 1742),
        ("float32", [1025], False, 1743), ("float32", [1025], True, 1744),
        ("float32", [7, 9], False, 1745), ("float32", [7, 9], True, 1746)]):
    CORR.append((f"log2_-correctness-{i}", dt, shape, strided, True, seed))

TIMING = []
for i, (dt, shape, seed) in enumerate([
        ("float16", [1024], 1729), ("float16", [65536], 1730),
        ("float16", [1048576], 1731), ("float16", [4096, 4096], 1732),
        ("bfloat16", [1024], 1733), ("bfloat16", [65536], 1734),
        ("bfloat16", [1048576], 1735), ("bfloat16", [4096, 4096], 1736),
        ("float32", [1024], 1737), ("float32", [65536], 1738),
        ("float32", [1048576], 1739), ("float32", [4096, 4096], 1740)]):
    TIMING.append((f"log2_-timing-{i}", dt, shape, False, False, seed))

EDGE = [0.0, -0.0, float("nan"), float("inf"), -float("inf"), -1.0, 1.0]


def gen(dtype, shape, strided, special, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g)
    x = x.abs() + 0.05
    x = x.to(device=DEV, dtype=getattr(torch, dtype))
    if special and x.numel() >= 7:
        x.flatten()[:7] = torch.tensor(EDGE, dtype=x.dtype, device=DEV)
    base = None
    if strided:
        base = torch.empty((*x.shape, 2), dtype=x.dtype, device=DEV)
        base[..., 1] = 12345.0
        base[..., 0] = x
        x = base[..., 0]
    return x, base


def compare(out, ref, rtol, atol):
    o = out.detach().to(torch.float32)
    r = ref.detach().to(torch.float32)
    nan_o, nan_r = torch.isnan(o), torch.isnan(r)
    both_nan = nan_o & nan_r
    close = torch.abs(o - r) <= atol + rtol * torch.abs(r)
    inv_ok = (o == r) | (nan_o & nan_r)          # covers +-inf exact equality
    ok = (close & ~nan_r) | both_nan | (inv_ok & ~close)
    bad = ~ok
    nbad = int(bad.sum().item())
    detail = ""
    if nbad:
        idx = bad.flatten().nonzero()[:5].flatten().tolist()
        fo = o.flatten()
        fr = r.flatten()
        pairs = [(int(i), float(fo[i]), float(fr[i])) for i in idx]
        detail = f" first_bad(idx, cand, ref)={pairs}"
    return nbad, detail


def main():
    res = {"cases": [], "timing": {}, "micro": {}, "notes": []}
    spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
    cand = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cand)

    for name, dtype, shape, strided, special, seed in CORR + TIMING:
        try:
            A, base = gen(dtype, shape, strided, special, seed)
            rtol, atol = ((1e-4, 1e-4) if dtype == "float32" else (1.5e-2, 1.5e-2))
            ref = A.detach().clone()
            ref = ref.contiguous().clone()
            ref.log2_()
            pad_before = None
            if base is not None:
                pad_before = base[..., 1].clone()
            ptr_before = A.data_ptr()
            stride_before = A.stride()
            ret = cand.run(A)
            torch.npu.synchronize()
            nbad, detail = compare(A, ref, rtol, atol)
            entry = {
                "case": name, "dtype": dtype, "shape": list(shape),
                "strided": strided, "special": special,
                "returned_same_object": ret is A,
                "data_ptr_unchanged": A.data_ptr() == ptr_before,
                "stride_unchanged": A.stride() == stride_before,
                "out_dtype_ok": A.dtype == getattr(torch, dtype),
                "mismatch": nbad, "detail": detail,
            }
            if pad_before is not None:
                entry["pad_untouched"] = bool(torch.equal(base[..., 1], pad_before))
            res["cases"].append(entry)
            print(name, "mismatch=", nbad, detail, flush=True)
        except Exception:
            res["cases"].append({"case": name, "error": traceback.format_exc()[-1200:]})
            print(name, "ERROR", flush=True)
            traceback.print_exc()

    # ---- host-cost micro measurements -------------------------------------
    def bench(fn, iters, trials=5):
        for _ in range(min(iters, 30)):
            fn()
        torch.npu.synchronize()
        ts = []
        for _ in range(trials):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.npu.synchronize()
            ts.append((time.perf_counter() - t0) / iters)
        ts.sort()
        return ts[len(ts) // 2] * 1e6

    try:
        A = torch.rand(1024, device=DEV, dtype=torch.float16) + 0.5
        B = torch.rand(1024, device=DEV, dtype=torch.float16) + 0.5
        fp = A.data_ptr
        res["micro"]["data_ptr"] = bench(lambda: fp(), 2000)
        res["micro"]["get_device"] = bench(lambda: A.get_device(), 2000)
        res["micro"]["is_contiguous"] = bench(lambda: A.is_contiguous(), 2000)
        res["micro"]["npu_current_stream_obj"] = bench(
            lambda: torch.npu.current_stream().npu_stream, 1000)
        try:
            import torch_npu
            raw = torch_npu._C._npu_getCurrentRawStreamNoWait
            res["micro"]["raw_stream"] = bench(lambda: raw(0), 2000)
        except Exception as exc:
            res["micro"]["raw_stream_err"] = repr(exc)
        res["micro"]["torch_log2_"] = bench(lambda: B.log2_(), 300)
        res["micro"]["cand_run"] = bench(lambda: cand.run(A), 300)
        # entry inventory (which block/kernel was used)
        inv = []
        for k, v in cand._ENTRIES.items():
            inv.append({"dtype": str(k[0]), "n": k[1], "dev": k[2],
                        "grid": list(v[3]), "block": v[4],
                        "direct": v[0] is not None,
                        "kernel": getattr(v[6], "__name__", str(v[6]))})
        res["entries"] = inv
    except Exception:
        res["notes"].append(traceback.format_exc()[-1500:])

    # ---- per size comparison ---------------------------------------------
    for name, dtype, shape, strided, special, seed in TIMING:
        try:
            A, _ = gen(dtype, shape, strided, special, seed)
            B, _ = gen(dtype, shape, strided, special, seed)
            n = A.numel()
            iters = 50 if n <= 1 << 20 else 20
            nat = bench(lambda: B.log2_(), iters)
            candt = bench(lambda: cand.run(A), iters)
            res["timing"][name] = {"native_us": nat, "cand_us": candt,
                                   "ratio": (nat / candt) if candt else None}
            print("timing", name, res["timing"][name], flush=True)
        except Exception:
            res["timing"][name] = {"error": traceback.format_exc()[-800:]}

    out = os.path.join(ART, "probe_log2.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    print("wrote", out, flush=True)
    bad = [c for c in res["cases"] if c.get("mismatch") or c.get("error")
           or not c.get("returned_same_object", True)]
    print("FAILED_CASES", len(bad), flush=True)


main()
