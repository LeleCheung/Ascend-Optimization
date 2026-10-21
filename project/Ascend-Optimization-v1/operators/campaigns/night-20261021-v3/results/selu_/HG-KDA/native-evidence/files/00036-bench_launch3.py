"""Diagnostic: find a low-overhead host launch path for a @triton.jit kernel."""
import inspect
import json
import os
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


def bench_percall(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return round(statistics.median(ts), 2)


def main():
    dev = torch.npu.current_device()
    # device properties through the triton driver (authoritative for grid sizing)
    try:
        from triton.runtime import driver as tdrv
        props = tdrv.active.get_current_device()
        OUT["current_device"] = props
        try:
            gp = tdrv.active.utils.get_device_properties(dev)
            OUT["driver_props"] = {k: v for k, v in dict(gp).items()
                                   if isinstance(v, (int, float, str))}
        except Exception as e:  # noqa
            OUT["driver_props"] = f"ERR {type(e).__name__}: {e}"
    except Exception as e:  # noqa
        OUT["driver_props"] = f"ERR {type(e).__name__}: {e}"

    # --- inspect triton internals we may lean on ---
    from triton.runtime.jit import JITFunction
    try:
        src = inspect.getsource(JITFunction.run)
        OUT["jit_run_src"] = src[:4000]
    except Exception as e:  # noqa
        OUT["jit_run_src"] = f"ERR {e}"
    try:
        from triton.compiler.compiler import CompiledKernel
        OUT["ck_getitem_src"] = inspect.getsource(CompiledKernel.__getitem__)[:2500]
        OUT["ck_launch_src"] = inspect.getsource(CompiledKernel.launch_metadata)[:1500]
        OUT["ck_run_doc"] = str(getattr(CompiledKernel.run, "__doc__", ""))[:800]
    except Exception as e:  # noqa
        OUT["ck_getitem_src"] = f"ERR {e}"

    res = {}
    x = torch.randn(1024, device=f"npu:{dev}", dtype=torch.float16)
    n = x.numel()

    res["torch_batch"] = bench_batch(lambda: torch.selu_(x))
    res["triton_batch"] = bench_batch(lambda: selu_kernel_[(1,)](x, n, BLOCK_SIZE=1024))

    # --- flag_gems libentry applied the documented way: @libentry() over JITFunction ---
    try:
        from flag_gems.utils.libentry import libentry
        kefn = libentry()(selu_kernel_)
        OUT["libentry_type"] = type(kefn).__name__
        kefn[(1,)](x, n, BLOCK_SIZE=1024)
        torch.npu.synchronize()
        res["libentry_batch"] = bench_batch(lambda: kefn[(1,)](x, n, BLOCK_SIZE=1024))
        res["libentry_percall"] = bench_percall(lambda: kefn[(1,)](x, n, BLOCK_SIZE=1024))
        # what does LibEntry.run do?
        try:
            OUT["libentry_run_src"] = inspect.getsource(type(kefn).__mro__[0])[:3000]
        except Exception as e:  # noqa
            OUT["libentry_run_src"] = f"ERR {e}"
    except Exception as e:  # noqa
        OUT["libentry"] = f"ERR {type(e).__name__}: {str(e)[:600]}"

    # --- hand-rolled cached launcher on top of the compiled kernel ---
    try:
        from triton.runtime import driver as tdrv
        selu_kernel_[(1,)](x, n, BLOCK_SIZE=1024)  # populate device cache
        cache = selu_kernel_.device_caches
        dev_key = list(cache.keys())[0]
        kern = list(cache[dev_key][0].values())[0]
        OUT["compiled_kernel_type"] = type(kern).__name__
        nargs = kern.run.__doc__
        OUT["ck_run_sig"] = str(nargs)[:400]

        stream_get = tdrv.active.get_current_stream
        meta = kern.packed_metadata
        fn = kern.function
        run = kern.run
        grid0 = 1

        def hand(xx, nn):
            stream = stream_get(0)
            run(grid0, 1, 1, stream, fn, meta, None, None, None, xx, nn)

        try:
            hand(x, n)
            torch.npu.synchronize()
            res["hand_batch"] = bench_batch(lambda: hand(x, n))
            res["hand_percall"] = bench_percall(lambda: hand(x, n))
            res["hand_ok"] = True
        except Exception as e:  # noqa
            OUT["hand_err"] = f"{type(e).__name__}: {str(e)[:600]}"
    except Exception as e:  # noqa
        OUT["hand_outer_err"] = f"{type(e).__name__}: {str(e)[:600]}"

    OUT["results_us"] = res
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "launch_bench3.json"), "w") as fh:
        json.dump(OUT, fh, indent=2, default=str)
    print(json.dumps(OUT["results_us"], indent=2, default=str))
    print("---- jit_run_src ----")
    print(OUT.get("jit_run_src", "")[:3000])
    print("---- ck_getitem ----")
    print(OUT.get("ck_getitem_src", "")[:2000])


if __name__ == "__main__":
    main()
