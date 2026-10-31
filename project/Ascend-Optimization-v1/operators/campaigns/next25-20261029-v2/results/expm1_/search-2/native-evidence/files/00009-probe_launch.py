"""Bounded target probe: where does the per-call cost of an elementwise Triton
launch on Ascend actually go, and how does tile size / masking change it?

Measures, for a few representative workloads:
  submit_us        : host cost per launch with NO synchronisation inside the loop
                     (the NPU queue absorbs device work, so this is the CPU-side
                     dispatch + launch-enqueue cost)
  pipeline_us      : total walltime / iters for a loop without per-iteration sync
  sync_each_us     : total walltime / iters when every iteration synchronises
                     (what a per-call walltime timer would see)
  torch_sync_each  : same for torch's in-place expm1_ (host-cost floor reference)

Emits artifacts/probe.json.
"""

import json
import os
import time
import traceback

import torch

try:
    import torch_npu  # noqa: F401
except Exception:  # pragma: no cover
    pass

import triton
import triton.language as tl

OUT = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
DEV = "npu:0"
ITERS = 30


@triton.jit
def flat_masked(a_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(a_ptr + offs, mask=m, other=0.0)
    y = tl.exp(x.to(tl.float32)) - 1.0
    tl.store(a_ptr + offs, y.to(a_ptr.dtype.element_ty), mask=m)


@triton.jit
def flat_nomask(a_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a_ptr + offs)
    y = tl.exp(x.to(tl.float32)) - 1.0
    tl.store(a_ptr + offs, y.to(a_ptr.dtype.element_ty))


def bench(fn):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    t2 = time.perf_counter()
    t3 = time.perf_counter()
    for _ in range(ITERS):
        fn()
        torch.npu.synchronize()
    t4 = time.perf_counter()
    return {
        "submit_us": (t1 - t0) / ITERS * 1e6,
        "pipeline_us": (t2 - t0) / ITERS * 1e6,
        "sync_each_us": (t4 - t3) / ITERS * 1e6,
    }


def main():
    info = {
        "torch": torch.__version__,
        "triton": triton.__version__,
    }
    try:
        props = torch.npu.get_device_properties(0)
        info["device"] = {
            k: getattr(props, k)
            for k in ("name", "num_aicore", "num_vectorcore", "total_memory")
            if hasattr(props, k)
        }
    except Exception:
        info["device"] = traceback.format_exc(limit=1)

    cases = [
        ("f16_1024", torch.float16, 1024),
        ("f32_1024", torch.float32, 1024),
        ("f32_1M", torch.float32, 1048576),
        ("f16_16M", torch.float16, 4096 * 4096),
        ("f32_16M", torch.float32, 4096 * 4096),
    ]
    variants = [
        ("masked512", flat_masked, 512, True),
        ("masked4096", flat_masked, 4096, True),
        ("nomask4096", flat_nomask, 4096, False),
    ]

    results = {}
    for cname, dtype, n in cases:
        x = torch.randn(n, dtype=torch.float32, device=DEV).to(dtype)
        entry = {}

        def torch_call(x=x):
            x.expm1_()

        with torch.no_grad():
            try:
                entry["torch_inplace"] = bench(torch_call)
            except Exception:
                entry["torch_inplace"] = traceback.format_exc(limit=1)

            for vname, kern, block, masked in variants:
                grid = ((n + block - 1) // block,)

                if masked:
                    def call(kern=kern, grid=grid, block=block, x=x, n=n):
                        kern[grid](x, n, BLOCK=block)
                else:
                    if n % block:
                        continue

                    def call(kern=kern, grid=grid, block=block, x=x, n=n):
                        kern[grid](x, n, BLOCK=block)

                try:
                    entry[vname] = bench(call)
                except Exception:
                    entry[vname] = traceback.format_exc(limit=1)

        # numeric sanity: fp32 16M only, compare to reference on fresh data
        results[cname] = entry

    # accuracy screening of the exact kernel math on special values
    try:
        edge = torch.tensor([0.0, -0.0, float("nan"), float("inf"), -float("inf"),
                             -1.0, 1.0, 1e-6, -1e-6, 1e-3, 0.5, -5.0, 5.0],
                            dtype=torch.float32, device=DEV)
        ref = torch.expm1(edge)
        got = edge.clone()
        flat_masked[(1,)](got, got.numel(), BLOCK=16)
        acc = {
            "ref": [None if v != v else float(v) for v in ref.tolist()],
            "got": [None if v != v else float(v) for v in got.tolist()],
            "max_abs_diff": float((ref - got).abs().nan_to_num(nan=0.0).max()),
        }
        for dt in (torch.float16, torch.bfloat16):
            e2 = edge.to(dt)
            r2 = torch.expm1(e2).to(torch.float32)
            g2 = e2.clone()
            flat_masked[(1,)](g2, g2.numel(), BLOCK=16)
            acc[str(dt)] = {
                "max_abs_diff": float((r2 - g2.to(torch.float32)).abs()
                                      .nan_to_num(nan=0.0).max()),
                "got": [None if v != v else float(v) for v in g2.to(torch.float32).tolist()],
                "ref": [None if v != v else float(v) for v in r2.tolist()],
            }
        info["accuracy"] = acc
    except Exception:
        info["accuracy"] = traceback.format_exc(limit=3)

    info["results"] = results
    with open(os.path.join(OUT, "probe.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    print(json.dumps(info, indent=2)[:8000])


if __name__ == "__main__":
    main()
