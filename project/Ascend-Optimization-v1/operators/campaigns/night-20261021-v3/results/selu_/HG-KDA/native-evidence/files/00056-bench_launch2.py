"""Diagnostic: compare host launch paths and grid-stride tiling for selu_."""
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


@triton.jit
def selu_loop_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr, NPROG: tl.constexpr):
    pid = tl.program_id(axis=0)
    i = pid
    while i * BLOCK_SIZE < n_elements:
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        x_f32 = x.to(tl.float32)
        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        y_f32 = scale * tl.where(x_f32 > 0, x_f32, alpha * (tl.exp(x_f32) - 1.0))
        tl.store(x_ptr + offsets, y_f32.to(x.dtype), mask=mask)
        i += NPROG


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
    return {"median_us": round(statistics.median(ts), 2), "min_us": round(min(ts), 2)}


def bench_batch(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return {"per_call_us": round((time.perf_counter() - t0) * 1e6 / iters, 2)}


def main():
    dev = torch.npu.current_device()

    # --- what is LibEntry / libentry in this env? ---
    try:
        from flag_gems.utils.libentry import libentry
        OUT["libentry_module_src"] = inspect.getsource(
            inspect.getmodule(libentry))[:2000]
        deco = libentry()
        OUT["libentry_call"] = repr(deco)
    except Exception as e:  # noqa
        OUT["libentry_module_src"] = f"ERR {type(e).__name__}: {e}"

    try:
        props = torch.npu.get_device_properties(dev)
        OUT["num_vectorcore"] = getattr(props, "num_vectorcore", None)
        OUT["num_aicore"] = getattr(props, "num_aicore", None)
    except Exception as e:  # noqa
        OUT["props_err"] = str(e)
    try:
        from torch_npu.utils import _get_device_properties  # noqa
    except Exception:
        pass
    try:
        import torch_npu.npu.utils as nnu
        OUT["driver_props"] = "unused"
    except Exception:
        pass
    try:
        import torch_npu
        from torch_npu import _C  # noqa
        OUT["prop_try"] = "skip"
    except Exception:
        pass

    # --- host path variants on 1024 elements ---
    res = {}
    for name, shape, dt in [("fp16_1024", (1024,), torch.float16),
                            ("fp16_4096x4096", (4096, 4096), torch.float16)]:
        x = torch.randn(shape, device=f"npu:{dev}", dtype=dt)
        n = x.numel()
        r = {}
        r["torch_selu_"] = bench_percall(lambda: torch.selu_(x))
        r["torch_selu__batch"] = bench_batch(lambda: torch.selu_(x))

        for bs in (1024, 4096, 8192):
            g = (triton.cdiv(n, bs),)

            def f_plain(bs=bs, g=g):
                selu_kernel_[g](x, n, BLOCK_SIZE=bs)

            r[f"triton_bs{bs}"] = bench_percall(f_plain)
            r[f"triton_bs{bs}_batch"] = bench_batch(f_plain)

        # libentry-decorated wrapper
        try:
            from flag_gems.utils.libentry import libentry
            k = libentry()(selu_kernel_.fn)

            def f_lib():
                k[(triton.cdiv(n, 1024),)](x, n, BLOCK_SIZE=1024)

            f_lib()
            torch.npu.synchronize()
            ok = torch.allclose(x, torch.selu_(x.clone()), atol=0, rtol=0) if n < 4096 else True
            r["libentry_1024"] = bench_percall(f_lib)
            r["libentry_1024_batch"] = bench_batch(f_lib)
            r["libentry_ok"] = bool(ok)
        except Exception as e:  # noqa
            r["libentry_1024"] = f"ERR {type(e).__name__}: {e}"

        # direct compiled kernel handle
        try:
            ck = selu_kernel_.warmup(x, n, BLOCK_SIZE=1024, grid=(triton.cdiv(n, 1024), 1, 1))
            func = ck[(triton.cdiv(n, 1024), 1, 1)]

            def f_ck():
                func(x, n)

            r["ck_1024"] = bench_percall(f_ck)
            r["ck_1024_batch"] = bench_batch(f_ck)
        except Exception as e:  # noqa
            r["ck_1024"] = f"ERR {type(e).__name__}: {e}"

        res[name] = r

    # --- grid-stride loop kernel: how few programs are optimal? ---
    x = torch.randn((4096, 4096), device=f"npu:{dev}", dtype=torch.float16)
    n = x.numel()
    r = {}
    for npg in (20, 40, 80, 160, 320, 640):
        for bs in (4096, 8192):
            try:
                units = triton.cdiv(n, bs)
                gp = min(npg, units)

                def f(bs=bs, gp=gp):
                    selu_loop_kernel[(gp,)](x, n, BLOCK_SIZE=bs, NPROG=gp)

                r[f"loop_npg{npg}_bs{bs}"] = bench_percall(f, iters=50, warmup=10)
            except Exception as e:  # noqa
                r[f"loop_npg{npg}_bs{bs}"] = f"ERR {type(e).__name__}: {str(e)[:200]}"
    res["loop_4096x4096_fp16"] = r

    OUT["results"] = res
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "launch_bench2.json"), "w") as fh:
        json.dump(OUT, fh, indent=2, default=str)
    print(json.dumps({k: v for k, v in OUT.items() if k != "results"}, indent=2, default=str)[:3000])
    print(json.dumps(OUT["results"], indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
