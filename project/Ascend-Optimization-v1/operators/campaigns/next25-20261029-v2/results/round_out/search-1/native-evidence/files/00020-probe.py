# SPDX-License-Identifier: Apache-2.0
"""Bounded diagnostic: (1) full-domain numeric equivalence of round variants vs
the NPU torch.round oracle, (2) availability of single-instruction RNE
intrinsics, (3) relative timing on the three 4096x4096 timing workloads.

Artifacts: $KGS_DEBUG_ARTIFACTS/probe.json
"""
import json
import os
import sys
import time
import traceback

import torch
import torch_npu
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernels as K  # noqa: E402

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = {"env": {}, "api": {}, "numerics": {}, "timing": {}, "errors": []}


def note(key, val):
    OUT[key] = val


def probe_api():
    api = {}
    api["triton_version"] = getattr(triton, "__version__", "?")
    api["torch_version"] = torch.__version__
    api["torch_npu_version"] = getattr(torch_npu, "__version__", "?")
    names = {}
    for obj_name, obj in (("tl.math", getattr(tl, "math", None)),):
        if obj is None:
            names[obj_name] = None
            continue
        names[obj_name] = sorted(
            n for n in dir(obj)
            if any(k in n.lower() for k in ("rint", "round", "floor", "ceil", "trunc", "nearby"))
        )
    # candidate backend math modules
    import importlib
    for mod in ("triton.language.extra.cann.libdevice",
                "triton.language.extra.libdevice",
                "triton.language.extra.cann.extension",
                "triton.language.extra.cann",
                "triton.language.math"):
        try:
            m = importlib.import_module(mod)
            names[mod] = sorted(
                n for n in dir(m)
                if any(k in n.lower() for k in ("rint", "round", "floor", "ceil", "trunc", "nearby"))
            )
        except Exception as exc:  # noqa: BLE001
            names[mod] = "IMPORT_FAIL: %s" % type(exc).__name__
    api["symbols"] = names
    return api


def bits(dtype, n):
    """Full 16-bit domain for fp16/bf16 so every representable value is tested."""
    i = torch.arange(n, dtype=torch.int32).to(torch.int16)
    return i.view(dtype)


def cmp_oracle(name, fn, x, decimals, out_view=None):
    """Compare candidate against torch.round(..., out=) on the same input."""
    ref = torch.empty_like(x, memory_format=torch.contiguous_format) if out_view is None else out_view
    torch.round(x, decimals=decimals, out=ref)
    got = torch.empty_like(ref)
    fn(x, got, decimals)
    a, b = got.reshape(-1).float(), ref.reshape(-1).float()
    both_nan = torch.isnan(a) & torch.isnan(b)
    same_bits = got.reshape(-1).view(torch.int16 if got.element_size() == 2 else torch.int32) == \
        ref.reshape(-1).view(torch.int16 if ref.element_size() == 2 else torch.int32)
    ok = (a == b) | both_nan
    bad = (~ok).nonzero().flatten()
    diff = (a - b).abs()
    diff = torch.where(torch.isnan(diff), torch.zeros_like(diff), diff)
    rec = {
        "n": int(a.numel()),
        "mismatch": int(bad.numel()),
        "bitwise_mismatch": int((~same_bits).sum().item()),
        "max_abs_diff": float(diff.max().item()),
        "examples": [],
        "matched_ratio": 1.0 - float(bad.numel()) / max(a.numel(), 1),
    }
    for idx in bad[:8].tolist():
        rec["examples"].append([float(x.reshape(-1)[idx]), float(ref.reshape(-1)[idx]),
                                float(got.reshape(-1)[idx])])
    OUT["numerics"][name] = rec
    return rec


def launch(kern, x, out, decimals, block):
    n = x.numel()
    grid = ((n + block - 1) // block,)
    kern[grid](x, out, n, decimals, BLOCK=block)
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


def main():
    dev = torch.npu.current_device()
    props = torch.npu.get_device_properties(dev)
    note("env", {
        "device": props.name,
        "num_aicore": getattr(props, "num_aicore", None),
        "num_vectorcore": getattr(props, "num_vectorcore", None),
        "l2_cache_size": getattr(props, "l2_cache_size", None),
    })
    note("api", probe_api())

    # ---------------- numerics: full fp16 / bf16 domain, fp32 crafted grid ----
    for dtype in (torch.float16, torch.bfloat16):
        x = bits(dtype, 65536).npu()
        for dec in (0, 1):
            for vname, fn in (
                ("seed", lambda a, o, d: launch(K.k_seed, a, o, d, 1024)),
                ("magic_f32", lambda a, o, d: launch(K.k_magic_f32, a, o, d, 1024)),
                ("magic_dtype", lambda a, o, d: launch(K.k_magic_dtype, a, o, d, 1024)),
            ):
                try:
                    cmp_oracle("%s_%s_dec%d" % (vname, str(dtype).split('.')[-1], dec), fn, x, dec)
                except Exception:
                    OUT["errors"].append("numerics %s %s dec%d: %s"
                                         % (vname, dtype, dec, traceback.format_exc()[-400:]))

    # fp32: dense sweep around tie boundaries plus wide random values
    fp32_vals = []
    for base in (0.5, 1.5, 2.5, 8191.5, 4194303.5, 8388607.5, 0.25, 1.25, 0.05, 0.15):
        fp32_vals.append(torch.tensor([base], dtype=torch.float32) +
                         torch.linspace(-2e-3, 2e-3, 257, dtype=torch.float32))
        fp32_vals.append(torch.tensor([base], dtype=torch.float32) +
                         torch.linspace(-2e-3, 2e-3, 257, dtype=torch.float32) * 0 - 0.0)
    fp32_vals.append(torch.randn(1 << 20) * 10.0)
    fp32_vals.append(torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"),
                                   8388608.0, 16777216.0, 1e30, -1e30, 4.5e15], dtype=torch.float32))
    xf = torch.cat(fp32_vals).npu().to(torch.float32)
    for dec in (0, 1):
        for vname, fn in (
            ("seed", lambda a, o, d: launch(K.k_seed, a, o, d, 1024)),
            ("magic_f32", lambda a, o, d: launch(K.k_magic_f32, a, o, d, 1024)),
        ):
            try:
                cmp_oracle("%s_fp32_dec%d" % (vname, dec), fn, xf, dec)
            except Exception:
                OUT["errors"].append("numerics %s fp32 dec%d: %s"
                                     % (vname, dec, traceback.format_exc()[-400:]))

    # ---------------- timing on the three 4096x4096 workloads -----------------
    variants = (
        ("seed", K.k_seed),
        ("magic_f32", K.k_magic_f32),
        ("magic_dtype", K.k_magic_dtype),
        ("magic_f32_nomask", K.k_magic_f32_nomask),
        ("magic_dtype_nomask", K.k_magic_dtype_nomask),
    )
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        x = torch.randn(4096, 4096, dtype=torch.float32).to(dtype).npu()
        out = torch.empty_like(x)
        ref = torch.empty_like(x)
        row = {}
        row["torch_round_ms"] = bench(lambda: torch.round(x, decimals=0, out=ref))
        for vname, kern in variants:
            try:
                row[vname] = bench(lambda k=kern: launch(k, x, out, 0, 4096))
            except Exception:
                row[vname] = "FAIL: " + traceback.format_exc()[-200:]
        OUT["timing"]["4096x4096_" + str(dtype).split('.')[-1]] = row

    # ---------------- block sweep for the leading variant (fp16) --------------
    x = torch.randn(4096, 4096, dtype=torch.float32).to(torch.float16).npu()
    out = torch.empty_like(x)
    sweep = {}
    for block in (512, 1024, 2048, 4096, 8192):
        for vname, kern in (("magic_f32", K.k_magic_f32), ("magic_dtype", K.k_magic_dtype)):
            try:
                sweep["%s_b%d" % (vname, block)] = bench(lambda k=kern, b=block: launch(k, x, out, 0, b))
            except Exception:
                sweep["%s_b%d" % (vname, block)] = "FAIL: " + traceback.format_exc()[-160:]
    OUT["timing"]["block_sweep_fp16_4096x4096"] = sweep

    # ---------------- tiny-case host overhead (N=1024, fp16) -----------------
    x = torch.randn(1024, dtype=torch.float32).to(torch.float16).npu()
    out = torch.empty_like(x)
    tiny = {"torch_round_ms": bench(lambda: torch.round(x, decimals=0, out=out), iters=200)}
    for block in (256, 1024):
        for vname, kern in (("seed", K.k_seed), ("magic_f32", K.k_magic_f32),
                            ("magic_dtype", K.k_magic_dtype)):
            tiny["%s_b%d" % (vname, block)] = bench(
                lambda k=kern, b=block: launch(k, x, out, 0, b), iters=200)
    OUT["timing"]["tiny_1024_fp16"] = tiny

    with open(os.path.join(ART, "probe.json"), "w") as fh:
        json.dump(OUT, fh, indent=1, default=str)
    print(json.dumps({k: OUT[k] for k in ("env", "timing")}, indent=1, default=str))
    print("NUMERICS", json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "examples"} for k, v in OUT["numerics"].items()},
        indent=1))
    print("ERRORS", json.dumps(OUT["errors"], indent=1)[:3000])


if __name__ == "__main__":
    main()
