"""Microbenchmark host-side launch overhead variants for the selu_ kernel.

Diagnostic only: measures per-call walltime (launch + sync) for several host
dispatch paths and block sizes. Writes JSON to $KGS_DEBUG_ARTIFACTS.
"""
import json
import logging
import os
import statistics
import time

import torch
import triton
import triton.language as tl

OUT = {}
ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")


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


def bench(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)  # us
    return {"median_us": statistics.median(ts), "min_us": min(ts)}


def main():
    logger = logging.getLogger("probe")
    dev = torch.npu.current_device()
    OUT["device"] = str(dev)

    # --- environment facts ---
    try:
        import flag_gems
        OUT["flag_gems_version"] = getattr(flag_gems, "__version__", "?")
    except Exception as e:  # noqa
        OUT["flag_gems_version"] = f"ERR {e}"
    try:
        from flag_gems.utils.libentry import libentry
        OUT["libentry"] = "available"
    except Exception as e:  # noqa
        libentry = None
        OUT["libentry"] = f"ERR {e}"
    try:
        import inspect
        from flag_gems.utils import libentry as le_mod
        src = inspect.getsource(le_mod)
        OUT["libentry_src_len"] = len(src)
        OUT["libentry_has_cached"] = ("CachedKernelLauncher" in src) or ("cache" in src.lower())
        OUT["libentry_head"] = src[:1500]
    except Exception as e:  # noqa
        OUT["libentry_src"] = f"ERR {e}"

    try:
        import triton
        OUT["triton_version"] = triton.__version__
    except Exception:
        pass

    # --- variants ---
    def v_full(x, n):
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        with torch.npu.device(x.device):
            selu_kernel_[grid](x, n, BLOCK_SIZE=1024)

    def v_plain(x, n):
        selu_kernel_[(triton.cdiv(n, 1024),)](x, n, BLOCK_SIZE=1024)

    def v_plain_noctx(x, n):
        selu_kernel_[(triton.cdiv(n, 1024),)](x, n, BLOCK_SIZE=1024)

    variants = {
        "full(device_ctx+lambda+log)": None,
        "plain(tuple_grid)": None,
    }

    lib_mod = None
    if libentry is not None:
        mod = type(triton)
        try:
            k = libentry(selu_kernel_.fn)
        except Exception:
            try:
                k = libentry(selu_kernel_)
            except Exception as e:  # noqa
                k = None
                OUT["libentry_apply_err"] = str(e)
        if k is not None:
            lib_mod = k

    sizes = [
        ("fp16_1024", (1024,), torch.float16),
        ("fp16_1048576", (1048576,), torch.float16),
        ("fp16_4096x4096", (4096, 4096), torch.float16),
        ("fp32_4096x4096", (4096, 4096), torch.float32),
    ]

    results = {}
    for name, shape, dt in sizes:
        x = torch.randn(shape, device=f"npu:{dev}", dtype=dt)
        n = x.numel()
        r = {}

        r["torch_selu_"] = bench(lambda: torch.selu_(x))

        def f_full():
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
            selu_kernel_[grid](x, n, BLOCK_SIZE=1024)

        def f_plain():
            selu_kernel_[(triton.cdiv(n, 1024),)](x, n, BLOCK_SIZE=1024)

        r["triton_lambda"] = bench(f_full)
        r["triton_tuple"] = bench(f_plain)

        if lib_mod is not None:
            try:
                def f_lib():
                    lib_mod[(triton.cdiv(n, 1024),)](x, n, BLOCK_SIZE=1024)
                r["libentry"] = bench(f_lib)
            except Exception as e:  # noqa
                r["libentry"] = f"ERR {type(e).__name__}: {e}"

        # block size sweep (plain tuple grid)
        for bs in (1024, 2048, 4096, 8192, 16384, 32768):
            try:
                def f_bs(bs=bs):
                    selu_kernel_[(triton.cdiv(n, bs),)](x, n, BLOCK_SIZE=bs)
                r[f"bs{bs}"] = bench(f_bs, iters=60, warmup=10)
            except Exception as e:  # noqa
                r[f"bs{bs}"] = f"ERR {type(e).__name__}: {e}"
        results[name] = r

    OUT["results"] = results

    # direct compiled-kernel handle probe (no timing, just API availability)
    try:
        x = torch.randn(1024, device=f"npu:{dev}", dtype=torch.float16)
        ck = selu_kernel_.warmup(x, 1024, BLOCK_SIZE=1024, grid=(1,))
        OUT["warmup_ok"] = type(ck).__name__
        try:
            ck[(1,)](x, 1024)
            OUT["compiled_launch"] = "ok"
        except Exception as e:  # noqa
            OUT["compiled_launch"] = f"ERR {type(e).__name__}: {e}"
    except Exception as e:  # noqa
        OUT["warmup_ok"] = f"ERR {type(e).__name__}: {e}"

    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "launch_bench.json"), "w") as fh:
        json.dump(OUT, fh, indent=2, default=str)
    print(json.dumps(OUT, indent=2, default=str)[:6000])


if __name__ == "__main__":
    main()
