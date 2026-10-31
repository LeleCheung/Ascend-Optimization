"""Bounded diagnosis: (a) which elements diverge for the 3 failing strided
correctness cases, (b) where the per-call wall time goes for tiny vs large
inputs, (c) device properties.  Diagnostics only -- never a formal result.
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

import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[1]          # .../tmp
sys.path.insert(0, str(ROOT))
import main as cand                                  # noqa: E402

ART = Path(os.environ.get("KGS_DEBUG_ARTIFACTS", ".")) / "probe1.json"
DEV = "npu:0"
EDGE = [0., -0., float("nan"), float("inf"), -float("inf"), -1., 1.]


def fmt(v):
    t = torch.tensor([v])
    return {"f": repr(float(v)), "bits": int(t.view(torch.int32).item())}


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


@triton.jit
def _neg_min(X, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    v = tl.load(X + o, mask=m)
    tl.store(X + o, -v, mask=m)


@triton.jit
def _noop(X):
    pid = tl.program_id(0)


def bench(fn, iters=200, sync_each=False, warmup=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    if sync_each:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
            torch.npu.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


CASES = [
    ("c5_fp16_7x9_strided", {"dtype": "float16", "shape": [7, 9], "strided": True, "special": True}, 1734),
    ("c11_bf16_7x9_strided", {"dtype": "bfloat16", "shape": [7, 9], "strided": True, "special": True}, 1740),
    ("c17_fp32_7x9_strided", {"dtype": "float32", "shape": [7, 9], "strided": True, "special": True}, 1746),
    ("c4_fp16_7x9_flat", {"dtype": "float16", "shape": [7, 9], "strided": False, "special": True}, 1733),
    ("c13_fp32_1_strided", {"dtype": "float32", "shape": [1], "strided": True, "special": True}, 1742),
    ("c15_fp32_1025_strided", {"dtype": "float32", "shape": [1025], "strided": True, "special": True}, 1744),
]

out = {"cases": {}, "timing": {}, "device": {}, "notes": []}


def main():
    # ---- correctness detail on the failing shapes -------------------------
    for name, spec, seed in CASES:
        try:
            a = gen(spec, seed)
            ref = a.clone().neg_()
            got = cand.run(a)
            torch.npu.synchronize()
            rf = torch.isfinite(ref)
            gf = torch.isfinite(got)
            diff = torch.nonzero(rf != gf).flatten().tolist()
            rec = {
                "spec": spec,
                "shape": list(got.shape),
                "stride": list(got.stride()),
                "ndim_after_gen": got.dim(),
                "collapse": cand._collapse(got),
                "finite_mismatch_idx": diff[:12],
                "n_finite_mismatch": len(diff),
                "n_nonfinite_ref": int((~rf).sum().item()),
                "n_nonfinite_got": int((~gf).sum().item()),
                # 0 == same object identity requirement
                "returned_is_input": got.data_ptr() == a.data_ptr(),
            }
            fl = got.flatten()
            rl = ref.flatten()
            rec["first7"] = [
                {"i": i, "ref": fmt(rl[i].item()), "got": fmt(fl[i].item())}
                for i in range(min(7, fl.numel()))
            ]
            ok = bool(torch.allclose(got.float(), ref.float(), rtol=1.5e-2, atol=1.5e-2, equal_nan=True))
            rec["allclose"] = ok
            out["cases"][name] = rec
        except Exception:
            out["cases"][name] = {"error": traceback.format_exc()}

    # ---- does the kernel touch storage outside the view? ------------------
    try:
        x = gen({"dtype": "float32", "shape": [7, 9], "strided": True, "special": True}, 1746)
        base = x._base
        if base is not None:
            base[..., 1] = 1234.0
            before = base[..., 1].clone()
            cand.run(x)
            torch.npu.synchronize()
            after = base[..., 1]
            out["outside_view_untouched"] = bool(torch.equal(before, after))
            out["outside_view_bad"] = int((before != after).sum().item())
    except Exception:
        out["outside_view_untouched"] = "error:" + traceback.format_exc()

    # ---- device properties ----------------------------------------------
    try:
        from triton.runtime import driver as tdrv
        props = tdrv.active.utils.get_device_properties(torch.npu.current_device())
        out["device"]["props"] = {str(k): (v if isinstance(v, (int, float, str, bool)) else str(v))
                                  for k, v in dict(props).items()}
    except Exception:
        out["device"]["props_error"] = traceback.format_exc()

    # ---- per-call wall time ---------------------------------------------
    shapes = [
        ("n1024_fp16", torch.zeros(1024, dtype=torch.float16, device=DEV), 1024),
        ("n1M_fp32", torch.zeros(1048576, dtype=torch.float32, device=DEV), 1048576),
        ("4096_fp32", torch.zeros((4096, 4096), dtype=torch.float32, device=DEV), 4096 * 4096),
        ("4096_fp16", torch.zeros((4096, 4096), dtype=torch.float16, device=DEV), 4096 * 4096),
    ]
    for name, a, n in shapes:
        t = {}
        t["torch_neg_"] = bench(lambda: a.neg_())
        t["cand_run"] = bench(lambda: cand.run(a))
        t["cand_run_sync_each"] = bench(lambda: cand.run(a), iters=50, sync_each=True)
        t["torch_neg_sync_each"] = bench(lambda: a.neg_(), iters=50, sync_each=True)
        b = 4096 if n > 4096 else max(1, 1 << max(0, (n - 1).bit_length()))
        grid = ((n + b - 1) // b,)
        t["min_kernel_2args"] = bench(lambda: _neg_min[grid](a, n, BLOCK=b))
        t["noop_1arg"] = bench(lambda: _noop[(1,)](a))
        t["python_only"] = bench(lambda: a.numel() + 0, iters=2000)
        out["timing"][name] = t
        out["notes"].append(f"{name}: n={n} block={b} grid={grid[0]}")

    try:
        c = torch.npu.current_stream()
        out["timing"]["current_stream_x1000_ms"] = bench(lambda: torch.npu.current_stream(), iters=1000)
    except Exception:
        pass


try:
    main()
except Exception:
    out["fatal"] = traceback.format_exc()
finally:
    ART.parent.mkdir(parents=True, exist_ok=True)
    ART.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str)[:12000])
