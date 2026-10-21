"""Diagnostic: verify ascend triton launcher arity/semantics for a cached direct launch."""
import json
import os
import re
import statistics
import time

import torch
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = {}


@triton.jit
def selu_kernel_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    y_f32 = scale * tl.where(x_f32 > 0, x_f32, alpha * (tl.exp(x_f32) - 1.0))
    tl.store(x_ptr + offsets, y_f32.to(x.dtype), mask=mask)


def bench_batch(fn, iters=300, warmup=30):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return round((time.perf_counter() - t0) * 1e6 / iters, 2)


def main():
    dev = torch.npu.current_device()

    # --- exact launcher parse format on the ascend backend ---
    try:
        p = "/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/driver.py"
        with open(p, "r", errors="ignore") as fh:
            lines = fh.read().splitlines()
        hits = []
        for i, ln in enumerate(lines, 1):
            if re.search(r"PyArg_ParseTuple|args_list|signature\s*=|def make_launcher|\"iii", ln):
                hits.append(f"{i}: {ln.strip()[:300]}")
        OUT["driver_parse_hits"] = hits[:40]
        # dump the region around the launcher parse call
        for i, ln in enumerate(lines, 1):
            if "PyArg_ParseTupleAndKeywords" in ln or "PyArg_ParseTuple" in ln:
                OUT["parse_region"] = "\n".join(
                    f"{j}: {lines[j-1]}" for j in range(max(1, i - 40), min(len(lines), i + 8)))
                break
    except Exception as e:  # noqa
        OUT["driver_grep_err"] = f"{type(e).__name__}: {e}"

    # --- build direct launcher from compiled kernel ---
    x0 = torch.randn(1024, device=f"npu:{dev}", dtype=torch.float16)
    selu_kernel_[(1,)](x0, 1024, BLOCK_SIZE=1024)
    dk = list(selu_kernel_.device_caches.keys())[0]
    kc, _kk, _tg, _be, binder = selu_kernel_.device_caches[dk]
    kern = list(kc.values())[0]
    OUT["kern_attrs"] = {k: str(getattr(kern, k))[:80] for k in
                         ("name", "packed_metadata", "num_warps") if hasattr(kern, k)}
    try:
        ba, spec, opts = binder(x0, 1024, BLOCK_SIZE=1024)
        OUT["bound_args_keys"] = [str(k) for k in ba.keys()] if hasattr(ba, "keys") else str(type(ba))
        OUT["bound_args_vals"] = [type(v).__name__ for v in ba.values()] if hasattr(ba, "values") else "?"
    except Exception as e:  # noqa
        OUT["binder_probe_err"] = f"{type(e).__name__}: {e}"

    from triton.runtime import driver as tdrv
    stream_get = tdrv.active.get_current_stream
    fn = kern.function
    meta = kern.packed_metadata
    run = kern.run

    def check(x, n, g, tail):
        """Direct launch; returns max abs diff vs torch reference on a copy."""
        ref = torch.selu(x.detach().clone())
        xc = x.detach().clone()
        run(g, 1, 1, stream_get(0), fn, meta, None, None, None, xc, n, tail)
        torch.npu.synchronize()
        d = (xc.float() - ref.float()).abs().max().item()
        return d

    # correctness of tail variants
    variants = {"tail0": 0, "tail_block": 1024, "tail_none": None}
    corr = {}
    for name, tail in variants.items():
        try:
            x = torch.randn((7, 9), device=f"npu:{dev}", dtype=torch.float32)
            corr[name] = check(x, 63, triton.cdiv(63, 1024), tail)
        except Exception as e:  # noqa
            corr[name] = f"ERR {type(e).__name__}: {str(e)[:150]}"
    OUT["tail_correctness_maxdiff_fp32_7x9"] = corr

    # also verify per-block-size (constexpr slot) for bs != 1024
    corr2 = {}
    for bs in (1024, 4096, 8192):
        x = torch.randn(100000, device=f"npu:{dev}", dtype=torch.float16)
        n = x.numel()
        g = triton.cdiv(n, bs)
        selu_kernel_[(g,)](x, n, BLOCK_SIZE=bs)  # compile+populate
        torch.npu.synchronize()
        kc2 = selu_kernel_.device_caches[list(selu_kernel_.device_caches.keys())[0]][0]
        k2 = None
        for c in kc2.values():
            md = getattr(c, "metadata", None)
            if md is not None and getattr(md, "name", "").startswith("selu_kernel_"):
                if getattr(md, "num_warps", None) is not None:
                    k2 = c
        # find by matching compiled constexpr: use the one whose metadata matches bs
        cand = [c for c in kc2.values()
                if getattr(getattr(c, "metadata", None), "name", "").startswith("selu_kernel_")]
        ok = {}
        for c in cand:
            try:
                dd = check(x, n, g, 0) if c is kern else None
            except Exception:
                dd = None
        # generic: use the most recently inserted kernel matching bs via warmup call
        ck = selu_kernel_.warmup(x, n, BLOCK_SIZE=bs,
                                 grid=(g, 1, 1))
        try:
            ref = torch.selu(x.detach().clone())
            xc = x.detach().clone()
            ck.run(g, 1, 1, tdrv.active.get_current_stream(0), ck.function,
                   ck.packed_metadata, None, None, None, xc, n, 0)
            torch.npu.synchronize()
            corr2[f"bs{bs}_tail0"] = (xc.float() - ref.float()).abs().max().item()
        except Exception as e:  # noqa
            corr2[f"bs{bs}_tail0"] = f"ERR {type(e).__name__}: {str(e)[:120]}"
    OUT["constexpr_tail_correctness"] = corr2

    # --- timing with tail=0 on cheap path ---
    res = {}
    for label, shape, dt in [("fp16_1024", (1024,), torch.float16),
                             ("fp16_4096x4096", (4096, 4096), torch.float16)]:
        x = torch.randn(shape, device=f"npu:{dev}", dtype=dt)
        n = x.numel()
        r = {"torch": bench_batch(lambda: torch.selu_(x))}
        for bs in (1024, 8192):
            g = triton.cdiv(n, bs)
            ck = selu_kernel_.warmup(x, n, BLOCK_SIZE=bs, grid=(g, 1, 1))
            f, m, rr = ck.function, ck.packed_metadata, ck.run

            def direct(x=x, n=n, g=g, f=f, m=m, rr=rr):
                rr(g, 1, 1, stream_get(0), f, m, None, None, None, x, n, 0)

            r[f"direct_bs{bs}"] = bench_batch(direct)

            def viajit(x=x, n=n, g=g, bs=bs):
                selu_kernel_[(g,)](x, n, BLOCK_SIZE=bs)

            r[f"jit_bs{bs}"] = bench_batch(viajit)
        res[label] = r
    OUT["timing_us"] = res

    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "launch_bench5.json"), "w") as fh:
        json.dump(OUT, fh, indent=2, default=str)
    print(json.dumps({k: v for k, v in OUT.items() if k != "parse_region"}, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
