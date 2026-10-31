# SPDX-License-Identifier: Apache-2.0
"""Bounded diagnostic (<=120s): numeric equivalence of the magic-constant RNE
variants against the NPU torch.round oracle, plus relative timing of the two
leading variants on the 4096x4096 timing workloads.

Writes $KGS_DEBUG_ARTIFACTS/probe.json incrementally.
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
OUT = {"env": {}, "numerics": {}, "timing": {}, "errors": []}


def flush():
    with open(os.path.join(ART, "probe.json"), "w") as fh:
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
    both_nan = torch.isnan(a) & torch.isnan(b)
    ok = (a == b) | both_nan
    idt = torch.int16 if got.element_size() == 2 else torch.int32
    same_bits = got.reshape(-1).view(idt) == ref.reshape(-1).view(idt)
    bad = (~ok).nonzero().flatten()
    diff = torch.where(torch.isnan(a - b), torch.zeros_like(a), (a - b).abs())
    rec = {"n": int(a.numel()), "mismatch": int(bad.numel()),
           "bitwise_mismatch": int((~same_bits).sum().item()),
           "max_abs_diff": float(diff.max().item()),
           "matched_ratio": 1.0 - float(bad.numel()) / max(a.numel(), 1),
           "examples": [[float(x.reshape(-1)[i]), float(ref.reshape(-1)[i]), float(got.reshape(-1)[i])]
                        for i in bad[:8].tolist()]}
    OUT["numerics"][name] = rec
    flush()
    return rec


def bits(dtype, n):
    return torch.arange(n, dtype=torch.int32).to(torch.int16).view(dtype)


def main():
    props = torch.npu.get_device_properties(torch.npu.current_device())
    OUT["env"] = {"device": props.name,
                  "num_aicore": getattr(props, "num_aicore", None),
                  "num_vectorcore": getattr(props, "num_vectorcore", None),
                  "l2_cache_size": getattr(props, "l2_cache_size", None),
                  "triton": getattr(triton, "__version__", "?"),
                  "torch": torch.__version__}
    flush()

    # ---- numerics: full 16-bit domain (every representable fp16/bf16 value) --
    for dtype in (torch.float16, torch.bfloat16):
        tag = str(dtype).split('.')[-1]
        x = bits(dtype, 65536).npu()
        for vname, kern in (("magic_f32", K.k_magic_f32), ("magic_dtype", K.k_magic_dtype)):
            try:
                cmp_oracle("%s_%s_dec0" % (vname, tag),
                           lambda a, o, d, k=kern: launch(k, a, o, d, 1024), x, 0)
            except Exception:
                OUT["errors"].append("%s %s dec0: %s" % (vname, tag, traceback.format_exc()[-300:]))
        if dtype is torch.float16:
            try:
                cmp_oracle("magic_f32_%s_dec1" % tag,
                           lambda a, o, d: launch(K.k_magic_f32, a, o, d, 1024), x, 1)
            except Exception:
                OUT["errors"].append("magic_f32 %s dec1: %s" % (tag, traceback.format_exc()[-300:]))

    # ---- numerics: fp32 tie grid + wide random + specials -------------------
    parts = []
    for base in (0.5, 1.5, 2.5, 8191.5, 4194303.5, 8388607.5, 0.25, 1.25):
        parts.append(torch.tensor([base], dtype=torch.float32)
                     + torch.linspace(-2e-3, 2e-3, 257, dtype=torch.float32))
    parts.append(torch.randn(1 << 18) * 10.0)
    parts.append(torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"),
                               8388608.0, 16777216.0, 1e30, -1e30, 4.5e15], dtype=torch.float32))
    xf = torch.cat(parts).npu().to(torch.float32)
    for vname, kern in (("magic_f32", K.k_magic_f32), ("seed", K.k_seed)):
        try:
            cmp_oracle("%s_fp32_dec0" % vname,
                       lambda a, o, d, k=kern: launch(k, a, o, d, 1024), xf, 0)
        except Exception:
            OUT["errors"].append("%s fp32 dec0: %s" % (vname, traceback.format_exc()[-300:]))
    try:
        cmp_oracle("magic_f32_fp32_dec1",
                   lambda a, o, d: launch(K.k_magic_f32, a, o, d, 1024), xf, 1)
    except Exception:
        OUT["errors"].append("magic_f32 fp32 dec1: %s" % traceback.format_exc()[-300:])

    # ---- timing: 4096x4096, BLOCK=4096 -------------------------------------
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        tag = str(dtype).split('.')[-1]
        x = torch.randn(4096, 4096, dtype=torch.float32).to(dtype).npu()
        out = torch.empty_like(x)
        row = {"torch_round_ms": bench(lambda: torch.round(x, decimals=0, out=out))}
        for vname, kern in (("magic_f32", K.k_magic_f32), ("magic_dtype", K.k_magic_dtype)):
            try:
                row[vname] = bench(lambda k=kern: launch(k, x, out, 0, 4096))
            except Exception:
                row[vname] = "FAIL: " + traceback.format_exc()[-160:]
        OUT["timing"]["4096x4096_" + tag] = row
        flush()

    with open(os.path.join(ART, "probe_done.json"), "w") as fh:
        json.dump({"numerics": OUT["numerics"], "timing": OUT["timing"]}, fh, indent=1, default=str)

    print("NUMERICS", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "examples"}
                                  for k, v in OUT["numerics"].items()}, indent=1))
    print("TIMING", json.dumps(OUT["timing"], indent=1, default=str))
    print("ERRORS", json.dumps(OUT["errors"], indent=1)[:2000])


if __name__ == "__main__":
    main()
