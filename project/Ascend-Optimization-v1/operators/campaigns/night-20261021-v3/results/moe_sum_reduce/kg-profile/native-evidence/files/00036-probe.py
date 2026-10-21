"""Diagnostic probe of the candidate wrapper in tmp/main.py (diagnostic only)."""
import json
import os
import sys
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, "tmp")
import main as cand  # noqa: E402

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = {"wrapper": {}, "ref": {}, "override": {}, "correct": {}}
T0 = time.time()


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


def make(T, K=8, H=7168):
    x = torch.randn(T, K, H, device="npu", dtype=torch.float32).to(torch.bfloat16)
    return x


K_ = cand._moe_sum_reduce_contig_kernel

TIMING = [(1, 8, 7168), (8, 8, 7168), (64, 8, 7168), (512, 8, 7168), (4096, 8, 7168)]

# ---------------------------------------------------------------- wrapper timing
for T, K, H in TIMING:
    x = make(T, K, H)
    out = cand.run(x, 2.5)
    t_ms = bench(lambda: cand.run(x, 2.5), 30)
    rec("wrapper", f"T{T}", t_ms)
    rec("wrapper", f"T{T}_GBps", 9 * T * H * 2 / t_ms / 1e6)
    rec("ref", f"T{T}", bench(lambda: ref_fn(x, 2.5), 30))
    del x, out

# ---------------------------------------------------------------- geometry overrides
OVR = [(16, 1024, 4, 2), (32, 1024, 4, 2), (32, 512, 4, 2), (8, 1024, 4, 2),
       (64, 512, 4, 2), (16, 1024, 4, 1), (16, 2048, 4, 2), (16, 1024, 4, 4),
       (16, 1024, 8, 2), (24, 1024, 4, 2)]
for T in (4096, 512, 64):
    x = make(T, 8, 7168)
    outp = torch.empty(T, 7168, device="npu", dtype=torch.bfloat16)
    ref = ref_fn(x, 2.5)
    for bm, bn, w, s in OVR:
        if T % bm != 0:
            continue
        try:
            f = lambda: K_[((T + bm - 1) // bm, (7168 + bn - 1) // bn)](
                x, outp, 2.5, T=T, K=8, H=7168, BM=bm, BN=bn,
                EVEN_M=(T % bm == 0), EVEN_N=(7168 % bn == 0),
                num_warps=w, num_stages=s)
            outp.zero_()
            f()
            torch.npu.synchronize()
            err = (outp.float() - ref.float()).abs().max().item()
            t = bench(f, 20)
            rec("override", f"T{T}_bm{bm}_bn{bn}_w{w}_s{s}",
                {"ms": t, "GBps": 9 * T * 7168 * 2 / t / 1e6, "err": err})
        except Exception as exc:  # noqa
            rec("override", f"T{T}_bm{bm}_bn{bn}_w{w}_s{s}_ERROR", repr(exc)[:250])
    del x, outp, ref

# ---------------------------------------------------------------- correctness (all 8 shapes)
SHAPES = [(7, 2, 128), (83, 4, 512), (3, 8, 2048),
          (1, 8, 7168), (8, 8, 7168), (64, 8, 7168), (512, 8, 7168), (4096, 8, 7168)]
for T, K, H in SHAPES:
    x = make(T, K, H)
    ref = ref_fn(x, 2.5)
    act = cand.run(x, 2.5)
    a = act.float()
    r = ref.float()
    denom = r.abs().clamp_min(1e-30)
    rec("correct", f"T{T}_K{K}_H{H}", {
        "dtype_ok": act.dtype == ref.dtype,
        "shape_ok": tuple(act.shape) == tuple(ref.shape),
        "max_abs": (a - r).abs().max().item(),
        "max_rel": ((a - r).abs() / denom).max().item(),
        "nan": bool(torch.isnan(a).any().item()),
        "ref_absmax": r.abs().max().item(),
    })
    # non-contiguous input path
    if T >= 1:
        xt = make(H, T, K)
        xn = xt.permute(1, 2, 0)
        try:
            rn = ref_fn(xn, 2.5)
            an = cand.run(xn, 2.5)
            rec("correct", f"T{T}_K{K}_H{H}_strided", {
                "contig": xn.is_contiguous(),
                "max_abs": (an.float() - rn.float()).abs().max().item(),
            })
        except Exception as exc:  # noqa
            rec("correct", f"T{T}_K{K}_H{H}_strided_ERROR", repr(exc)[:250])
    del x, ref, act, a, r

OUT["elapsed_s"] = time.time() - T0
with open(os.path.join(ART, "probe.json"), "w") as fh:
    json.dump(OUT, fh, indent=1)
print("done", OUT["elapsed_s"], flush=True)
