"""Final validation of tmp/main.py before formal evaluation (diagnostic only)."""
import json
import os
import sys
import time

import torch

sys.path.insert(0, "tmp")
import main as cand  # noqa: E402

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = {"timing": {"cand": {}, "ref": {}}, "correct": {}, "fallback": {}}


def rec(bucket, key, val):
    OUT[bucket][key] = val
    print(bucket, key, val, flush=True)


def bench(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def ref_fn(x, s):
    return x.float().sum(dim=1).mul(s).to(x.dtype)


def make(T, K, H):
    return torch.randn(T, K, H, device="npu", dtype=torch.float32).to(torch.bfloat16)


SHAPES = [(7, 2, 128), (83, 4, 512), (3, 8, 2048),
          (1, 8, 7168), (8, 8, 7168), (64, 8, 7168), (512, 8, 7168), (4096, 8, 7168)]

for T, K, H in SHAPES:
    x = make(T, K, H)
    ref = ref_fn(x, 2.5)
    act = cand.run(x, 2.5)
    rec("correct", f"T{T}_K{K}_H{H}", {
        "dtype_ok": act.dtype == ref.dtype,
        "shape_ok": tuple(act.shape) == tuple(ref.shape),
        "max_abs": (act.float() - ref.float()).abs().max().item(),
        "nan": bool(torch.isnan(act.float()).any().item()),
    })
    if H == 7168:
        t = bench(lambda: cand.run(x, 2.5), 30)
        OUT["timing"]["cand"][f"T{T}"] = t
        OUT["timing"]["ref"][f"T{T}"] = bench(lambda: ref_fn(x, 2.5), 30)
        print("timing", T, t, OUT["timing"]["ref"][f"T{T}"], flush=True)
    # non-contiguous input -> generic fallback
    xt = make(H, T, K).permute(1, 2, 0) if T > 1 else None
    if xt is not None and not xt.is_contiguous():
        try:
            rn = ref_fn(xt, 2.5)
            an = cand.run(xt, 2.5)
            rec("fallback", f"T{T}_K{K}_H{H}", (an.float() - rn.float()).abs().max().item())
        except Exception as exc:  # noqa
            rec("fallback", f"T{T}_K{K}_H{H}_ERROR", repr(exc)[:200])
    del x, ref, act

# odd hidden_dim / odd token counts exercise the mask-free divisibility logic
for T, K, H in [(5, 4, 100), (17, 3, 384), (1, 1, 7168), (100, 2, 7168)]:
    x = make(T, K, H)
    ref = ref_fn(x, 2.5)
    try:
        act = cand.run(x, 2.5)
        rec("correct", f"odd_T{T}_K{K}_H{H}", {
            "shape_ok": tuple(act.shape) == tuple(ref.shape),
            "max_abs": (act.float() - ref.float()).abs().max().item(),
        })
    except Exception as exc:  # noqa
        rec("correct", f"odd_T{T}_K{K}_H{H}_ERROR", repr(exc)[:200])
    del x, ref

# empty tensor guard
xe = torch.empty(0, 8, 128, device="npu", dtype=torch.bfloat16)
try:
    oe = cand.run(xe, 2.5)
    rec("correct", "empty", list(oe.shape))
except Exception as exc:  # noqa
    rec("correct", "empty_ERROR", repr(exc)[:200])

with open(os.path.join(ART, "validate.json"), "w") as fh:
    json.dump(OUT, fh, indent=1)
print("done", flush=True)
