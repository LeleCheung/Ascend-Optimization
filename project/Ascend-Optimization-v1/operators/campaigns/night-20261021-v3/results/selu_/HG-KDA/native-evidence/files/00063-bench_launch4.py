"""Diagnostic: decompose triton per-call host cost and build a minimal launcher."""
import json
import os
import statistics
import time
import subprocess

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
    x = torch.randn(1024, device=f"npu:{dev}", dtype=torch.float16)
    n = x.numel()

    # find the generated launcher signature inside installed triton
    try:
        import triton as _t
        root = os.path.dirname(_t.__file__)
        OUT["triton_root"] = root
        hits = []
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dirpath, f)
                try:
                    with open(p, "r", errors="ignore") as fh:
                        txt = fh.read()
                except Exception:
                    continue
                if "launch_enter_hook" in txt:
                    for i, line in enumerate(txt.splitlines(), 1):
                        if ("global_scratch" in line or "profile_scratch" in line
                                or "launch_enter_hook" in line):
                            hits.append(f"{p}:{i}: {line.strip()[:200]}")
                if len(hits) > 60:
                    break
            if len(hits) > 60:
                break
        OUT["launcher_sig_hits"] = hits[:60]
    except Exception as e:  # noqa
        OUT["launcher_grep_err"] = str(e)

    # decomposition
    from triton.runtime import driver as tdrv
    decomp = {}
    try:
        decomp["get_current_stream"] = bench_batch(
            lambda: tdrv.active.get_current_stream(0), iters=2000, warmup=200)
    except Exception as e:  # noqa
        decomp["get_current_stream"] = f"ERR {e}"
    try:
        decomp["get_current_device"] = bench_batch(
            lambda: tdrv.active.get_current_device(), iters=2000, warmup=200)
    except Exception as e:  # noqa
        decomp["get_current_device"] = f"ERR {e}"

    selu_kernel_[(1,)](x, n, BLOCK_SIZE=1024)
    cache = selu_kernel_.device_caches
    dev_key = list(cache.keys())[0]
    kernel_cache, key_cache, target, backend, binder = cache[dev_key]
    kern = list(kernel_cache.values())[0]
    OUT["dir_kern"] = [a for a in dir(kern) if not a.startswith("_")][:60]

    try:
        decomp["binder"] = bench_batch(lambda: binder(x, n, BLOCK_SIZE=1024), iters=1000, warmup=100)
    except Exception as e:  # noqa
        decomp["binder"] = f"ERR {e}"
    try:
        from triton.runtime.jit import compute_cache_key
        ba, spec, opts = binder(x, n, BLOCK_SIZE=1024)
        decomp["compute_key"] = bench_batch(
            lambda: compute_cache_key(key_cache, spec, opts), iters=2000, warmup=200)
    except Exception as e:  # noqa
        decomp["compute_key"] = f"ERR {e}"

    # try fixed-arity direct launches: 9 fixed + 2 tensor args -> needs 12
    fn = kern.function
    meta = kern.packed_metadata
    run = kern.run
    stream_get = tdrv.active.get_current_stream
    fixed_tries = {}
    ok_call = None
    candidates = {
        "extra_before_args": lambda s: (1, 1, 1, s, fn, meta, None, None, None, 0, x, n),
        "extra_after_args": lambda s: (1, 1, 1, s, fn, meta, None, None, None, x, n, 0),
        "two_scratch_before": lambda s: (1, 1, 1, s, fn, meta, None, None, None, 0, 0, x, n),
        "scratch_after_exit": lambda s: (1, 1, 1, s, fn, meta, None, None, None, 0, x, n),
    }
    for name, mk in candidates.items():
        try:
            s = stream_get(0)
            run(*mk(s))
            torch.npu.synchronize()
            fixed_tries[name] = "ok"
            if ok_call is None:
                ok_call = name
        except Exception as e:  # noqa
            fixed_tries[name] = f"{type(e).__name__}: {str(e)[:150]}"
    OUT["fixed_tries"] = fixed_tries
    OUT["ok_call"] = ok_call

    res = {"torch_batch": bench_batch(lambda: torch.selu_(x))}
    if ok_call is not None:
        mk = candidates[ok_call]
        for label, (shape, dt) in {
            "fp16_1024": ((1024,), torch.float16),
            "fp16_4096x4096": ((4096, 4096), torch.float16),
        }.items():
            xx = torch.randn(shape, device=f"npu:{dev}", dtype=dt)
            nn = xx.numel()
            for bs in (1024, 8192):
                g = triton.cdiv(nn, bs)
                # warm compile for this bs
                selu_kernel_[(g,)](xx, nn, BLOCK_SIZE=bs)
                torch.npu.synchronize()
                kc, kkc, tg, be, bnd = selu_kernel_.device_caches[list(selu_kernel_.device_caches.keys())[0]]
                cands = list(kc.values())
                kk = None
                for c in cands:
                    if getattr(c, "metadata", None) is not None and c.metadata.name.startswith("selu_kernel_"):
                        kk = c
                if kk is None:
                    res[f"{label}_bs{bs}"] = "no kernel"
                    continue
                f2, m2, r2 = kk.function, kk.packed_metadata, kk.run

                def direct(xx=xx, nn=nn, g=g, s=stream_get, f2=f2, m2=m2, r2=r2, ok=ok_call):
                    st = s(0)
                    if ok == "extra_before_args":
                        r2(g, 1, 1, st, f2, m2, None, None, None, 0, xx, nn)
                    elif ok == "extra_after_args":
                        r2(g, 1, 1, st, f2, m2, None, None, None, xx, nn, 0)
                    else:
                        r2(g, 1, 1, st, f2, m2, None, None, None, 0, xx, nn)

                try:
                    res[f"{label}_bs{bs}_direct_batch"] = bench_batch(direct, iters=200, warmup=20)
                except Exception as e:  # noqa
                    res[f"{label}_bs{bs}_direct_batch"] = f"ERR {type(e).__name__}: {str(e)[:200]}"

                def viajit(xx=xx, nn=nn, g=g, bs=bs):
                    selu_kernel_[(g,)](xx, nn, BLOCK_SIZE=bs)

                res[f"{label}_bs{bs}_jit_batch"] = bench_batch(viajit, iters=200, warmup=20)

    OUT["decomp_us"] = decomp
    OUT["results_us"] = res
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "launch_bench4.json"), "w") as fh:
        json.dump(OUT, fh, indent=2, default=str)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
