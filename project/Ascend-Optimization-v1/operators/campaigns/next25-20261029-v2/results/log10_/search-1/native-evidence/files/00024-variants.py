"""Bounded device diagnostic for log10_ kernel structure on the large shapes.

Compares, on the real 4096x4096 inputs of the frozen workloads:
  * the current one-monolithic-tile-per-program kernel (BLOCK = UB cap)
  * an inner tl.range(num_stages=2/3) chunk loop over a larger outer tile
  * a small-tile / large-grid monolithic kernel
  * the log2-based math with the loop structure
  * the ATen in-place log10_ reference floor, measured in this same harness

Walltime per call (the evaluation's timing mode), warmup + repeats, and a
numerical check of every variant against torch.log10_ on fresh data.
"""

import json
import os
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
LOG10E = 0.4342944819032518
LOG10_2 = 0.3010299956639812


@triton.jit
def k_mono(x_ptr, BLOCK: tl.constexpr, USE_LOG2: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs).to(tl.float32)
    if USE_LOG2:
        y = tl.log2(x) * 0.3010299956639812
    else:
        y = tl.log(x) * 0.4342944819032518
    tl.store(x_ptr + offs, y.to(x_ptr.dtype.element_ty))


@triton.jit
def k_loop(x_ptr, BLOCK: tl.constexpr, CHUNK: tl.constexpr, ST: tl.constexpr,
           USE_LOG2: tl.constexpr):
    base = tl.program_id(0) * BLOCK
    for k in tl.range(0, BLOCK, CHUNK, num_stages=ST):
        offs = base + k + tl.arange(0, CHUNK)
        x = tl.load(x_ptr + offs).to(tl.float32)
        if USE_LOG2:
            y = tl.log2(x) * 0.3010299956639812
        else:
            y = tl.log(x) * 0.4342944819032518
        tl.store(x_ptr + offs, y.to(x_ptr.dtype.element_ty))


def bench(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # microseconds


def make_input(shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g).to(dtype=torch.float32)
    x = (x.abs() + 0.05).to(dtype)
    return x


def rel_err(buf, ref):
    a = buf.to(torch.float32)
    b = ref.to(torch.float32)
    denom = b.abs().clamp_min(1e-6)
    return float(((a - b).abs() / denom).max())


def run_shape(name, shape, dtype, seed, variants):
    base = make_input(shape, dtype, seed)
    n = base.numel()
    ref_out = torch.log10_(base.clone())
    esize = base.element_size()
    rows = []

    for label, build in variants:
        try:
            result = {"variant": label}
            launch = build(n, esize)
            # numerical check on fresh data
            work = base.clone()
            launch(work)
            torch.npu.synchronize()
            result["rel_err"] = rel_err(work, ref_out)
            result["nonfinite"] = int((~torch.isfinite(work.to(torch.float32))).sum().item())
            del work
            buf = base.clone()
            result["us"] = bench(lambda: launch(buf))
            result["logical_TBs"] = (2.0 * n * esize) / (result["us"] * 1e-6) / 1e12
            del buf
            rows.append(result)
        except Exception as exc:  # keep the other variants measurable
            rows.append({"variant": label, "error": repr(exc)[:300]})

    ref_buf = base.clone()
    ref_us = bench(lambda: torch.log10_(ref_buf))
    return {"workload": name, "shape": list(shape), "dtype": str(dtype),
            "n": n, "ref_us": ref_us,
            "ref_logical_TBs": (2.0 * n * base.element_size()) / (ref_us * 1e-6) / 1e12,
            "variants": rows}


def variants_for(cap2, cap4):
    def mono_cap(n, esize):
        block = cap2 if esize < 4 else cap4
        grid = (n // block,)
        return lambda t: k_mono[grid](t, BLOCK=block, USE_LOG2=False)

    def mono_small(n, esize):
        block = 2048
        grid = (n // block,)
        return lambda t: k_mono[grid](t, BLOCK=block, USE_LOG2=False)

    def loop2(n, esize):
        outer = 32768
        chunk = 4096 if esize < 4 else 2048
        grid = (n // outer,)
        return lambda t: k_loop[grid](t, BLOCK=outer, CHUNK=chunk, ST=2,
                                      USE_LOG2=False)

    def loop3(n, esize):
        outer = 32768
        chunk = 4096 if esize < 4 else 2048
        grid = (n // outer,)
        return lambda t: k_loop[grid](t, BLOCK=outer, CHUNK=chunk, ST=3,
                                      USE_LOG2=False)

    def loop2_log2(n, esize):
        outer = 32768
        chunk = 4096 if esize < 4 else 2048
        grid = (n // outer,)
        return lambda t: k_loop[grid](t, BLOCK=outer, CHUNK=chunk, ST=2,
                                      USE_LOG2=True)

    def mono_log2(n, esize):
        block = cap2 if esize < 4 else cap4
        grid = (n // block,)
        return lambda t: k_mono[grid](t, BLOCK=block, USE_LOG2=True)

    return [("mono_cap", mono_cap),
            ("mono_2048", mono_small),
            ("loop_outer32k_st2", loop2),
            ("loop_outer32k_st3", loop3),
            ("loop_outer32k_st2_log2", loop2_log2),
            ("mono_cap_log2", mono_log2)]


def main():
    out = {"cases": []}
    try:
        props = torch.npu.get_device_properties(torch.npu.current_device())
        out["device"] = {"name": props.name,
                         "num_vectorcore": getattr(props, "num_vectorcore", None),
                         "num_aicore": getattr(props, "num_aicore", None)}
    except Exception as exc:
        out["device"] = {"error": repr(exc)[:200]}

    out["cases"].append(run_shape("fp32_4096x4096", (4096, 4096), torch.float32,
                                  1740, variants_for(16384, 8192)))
    out["cases"].append(run_shape("fp16_4096x4096", (4096, 4096), torch.float16,
                                  1732, variants_for(16384, 8192)))

    os.makedirs(ART, exist_ok=True)
    path = os.path.join(ART, "log10_variants.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
