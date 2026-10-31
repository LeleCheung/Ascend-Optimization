# SPDX-License-Identifier: Apache-2.0
"""Bounded diagnostic 2 (<=120s): pure movement floor (copy kernel), availability
and cost of single-instruction RNE intrinsics, block-size sensitivity.
Writes $KGS_DEBUG_ARTIFACTS/probe2.json incrementally.
"""
import json
import os
import sys
import time
import traceback

import torch
import torch_npu
import triton

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernels as K  # noqa: E402

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = {"symbols": {}, "timing": {}, "numerics": {}, "errors": []}


def flush():
    with open(os.path.join(ART, "probe2.json"), "w") as fh:
        json.dump(OUT, fh, indent=1, default=str)


def launch(kern, x, out, decimals, block):
    n = x.numel()
    kern[((n + block - 1) // block,)](x, out, n, decimals, BLOCK=block)
    return out


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def cmp_oracle(name, fn, x, decimals):
    ref = torch.empty_like(x)
    torch.round(x, decimals=decimals, out=ref)
    got = torch.empty_like(x)
    fn(x, got, decimals)
    a, b = got.reshape(-1).float(), ref.reshape(-1).float()
    ok = (a == b) | (torch.isnan(a) & torch.isnan(b))
    bad = (~ok).nonzero().flatten()
    diff = torch.where(torch.isnan(a - b), torch.zeros_like(a), (a - b).abs())
    OUT["numerics"][name] = {
        "n": int(a.numel()), "mismatch": int(bad.numel()),
        "max_abs_diff": float(diff.max().item()),
        "examples": [[float(x.reshape(-1)[i]), float(ref.reshape(-1)[i]), float(got.reshape(-1)[i])]
                     for i in bad[:6].tolist()],
    }
    flush()
    return OUT["numerics"][name]


def bits(dtype, n):
    return torch.arange(n, dtype=torch.int32).to(torch.int16).view(dtype)


def main():
    OUT["symbols"] = {"tl.math": K.MATH_NAMES, "cann_libdevice": K.CLD_NAMES,
                      "have_cld": K.HAVE_CLD}
    flush()

    # ---------------- pure copy floor at 4096x4096, BLOCK=4096 ---------------
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        tag = str(dtype).split('.')[-1]
        x = torch.randn(4096, 4096, dtype=torch.float32).to(dtype).npu()
        out = torch.empty_like(x)
        row = {"torch_round_ms": bench(lambda: torch.round(x, decimals=0, out=out))}
        try:
            row["copy"] = bench(lambda: launch(K.k_copy, x, out, 0, 4096))
        except Exception:
            row["copy"] = "FAIL " + traceback.format_exc()[-160:]
        try:
            row["magic_f32"] = bench(lambda: launch(K.k_magic_f32, x, out, 0, 4096))
        except Exception:
            row["magic_f32"] = "FAIL " + traceback.format_exc()[-160:]
        OUT["timing"]["4096_" + tag] = row
        flush()

    # ---------------- RNE intrinsic candidates -------------------------------
    x16 = torch.randn(1 << 20, dtype=torch.float32).to(torch.float16).npu()
    x32 = torch.randn(1 << 20, dtype=torch.float32).npu()
    dom16 = bits(torch.float16, 65536).npu()
    for vname, kern in (("rint_cld", K.k_rint_cld), ("nearby_cld", K.k_nearby_cld),
                        ("rint_math", K.k_rint_math), ("round_math", K.k_round_math)):
        try:
            r = cmp_oracle(vname + "_fp16_domain", lambda a, o, d, k=kern: launch(k, a, o, d, 1024),
                           dom16, 0)
            OUT["symbols"].setdefault("compiled", []).append([vname, r["mismatch"]])
        except Exception:
            OUT["errors"].append("%s: %s" % (vname, traceback.format_exc()[-260:]))
        flush()

    # ---------------- timing of the survivors at 4096x4096 -------------------
    x = torch.randn(4096, 4096, dtype=torch.float32).to(torch.float16).npu()
    out = torch.empty_like(x)
    row = {}
    for vname, kern in (("copy", K.k_copy), ("magic_f32", K.k_magic_f32),
                        ("rint_cld", K.k_rint_cld), ("nearby_cld", K.k_nearby_cld),
                        ("rint_math", K.k_rint_math), ("round_math", K.k_round_math)):
        try:
            row[vname] = bench(lambda k=kern: launch(k, x, out, 0, 4096))
        except Exception:
            row[vname] = "FAIL " + traceback.format_exc()[-120:]
    OUT["timing"]["4096_float16_intrinsics"] = row
    flush()

    x = torch.randn(4096, 4096, dtype=torch.float32).npu()
    out = torch.empty_like(x)
    row = {"torch_round_ms": bench(lambda: torch.round(x, decimals=0, out=out))}
    for vname, kern in (("copy", K.k_copy), ("magic_f32", K.k_magic_f32),
                        ("rint_cld", K.k_rint_cld), ("rint_math", K.k_rint_math),
                        ("round_math", K.k_round_math)):
        try:
            row[vname] = bench(lambda k=kern: launch(k, x, out, 0, 4096))
        except Exception:
            row[vname] = "FAIL " + traceback.format_exc()[-120:]
    OUT["timing"]["4096_float32_intrinsics"] = row
    flush()

    # ---------------- block sensitivity (fp16) ------------------------------
    x = torch.randn(4096, 4096, dtype=torch.float32).to(torch.float16).npu()
    out = torch.empty_like(x)
    sweep = {}
    for block in (1024, 2048, 8192):
        for vname, kern in (("copy", K.k_copy), ("magic_f32", K.k_magic_f32)):
            try:
                sweep["%s_b%d" % (vname, block)] = bench(lambda k=kern, b=block: launch(k, x, out, 0, b))
            except Exception:
                sweep["%s_b%d" % (vname, block)] = "FAIL " + traceback.format_exc()[-120:]
    OUT["timing"]["block_sweep_fp16"] = sweep
    flush()

    print(json.dumps(OUT, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
