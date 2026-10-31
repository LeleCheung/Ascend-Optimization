"""Probe 2: (a) what does Triton's Python launch path on this Ascend build do
per call, and can a pre-resolved CompiledKernel be launched directly?
(b) how does device time for 16M elements scale with program count / tile size
(a loop kernel with a small grid vs one program per tile)?

Writes probe2.json (sources + timings) into $KGS_DEBUG_ARTIFACTS.
"""

import inspect
import json
import os
import time
import traceback

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
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
def loop_kernel(a_ptr, n, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    """Few programs, each striding over CHUNK-sized tiles."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    ntiles = (n + CHUNK - 1) // CHUNK
    for t in range(pid, ntiles, nprog):
        base = t * CHUNK + tl.arange(0, BLOCK)
        m = base < n
        x = tl.load(a_ptr + base, mask=m, other=0.0)
        y = tl.exp(x.to(tl.float32)) - 1.0
        tl.store(a_ptr + base, y.to(a_ptr.dtype.element_ty), mask=m)


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
    return {
        "submit_us": (t1 - t0) / ITERS * 1e6,
        "pipeline_us": (t2 - t0) / ITERS * 1e6,
    }


def main():
    info = {"torch": torch.__version__, "triton": triton.__version__}

    # ---- (a) sources of the launch path -------------------------------
    src = {}
    try:
        import triton.runtime.jit as _jit
        text = inspect.getsource(_jit.JITFunction.run)
        src["JITFunction.run"] = text
    except Exception:
        src["JITFunction.run"] = traceback.format_exc(limit=2)
    try:
        from triton.backends.ascend import driver as asc_driver
        src["ascend_driver_path"] = inspect.getsourcefile(asc_driver)
        text = inspect.getsource(asc_driver)
        src["ascend_driver"] = text[:60000]
    except Exception:
        src["ascend_driver"] = traceback.format_exc(limit=2)
    try:
        import triton.backends.ascend.driver as d
        names = [n for n in dir(d) if "aunch" in n or "ernel" in n]
        src["ascend_driver_symbols"] = names
    except Exception:
        pass
    info["sources"] = src

    # ---- (a2) direct CompiledKernel launch cost ----------------------
    direct = {}
    x = torch.randn(1048576, dtype=torch.float32, device=DEV)
    n = x.numel()
    grid = (n // 4096,)
    flat_masked[grid](x, n, BLOCK=4096)
    torch.npu.synchronize()
    try:
        ck = flat_masked.warmup(x, n, BLOCK=4096, grid=grid)
        direct["ck_type"] = str(type(ck))
        direct["ck_run_sig"] = str(inspect.signature(ck.run))
        direct["ck_attrs"] = [a for a in dir(ck) if not a.startswith("__")][:60]
        try:
            stream = torch.npu.current_stream()
            direct["stream_type"] = str(type(stream))
            direct["stream_repr"] = repr(stream)[:200]
        except Exception:
            direct["stream_type"] = traceback.format_exc(limit=1)
        # attempt the low level launch exactly like JITFunction.run does
        try:
            md = ck.packed_metadata
            direct["packed_metadata"] = repr(md)[:400]
        except Exception:
            direct["packed_metadata"] = traceback.format_exc(limit=1)
        for attempt in ("argstyle_a", "argstyle_b"):
            try:
                if attempt == "argstyle_a":
                    def call():
                        ck.run(*(grid + (1, 1)), stream, ck.function,
                               ck.packed_metadata, None, None, None, x, n)
                else:
                    def call():
                        ck[grid](x, n)
                direct[attempt] = bench(call)
            except Exception:
                direct[attempt] = traceback.format_exc(limit=3)
    except Exception:
        direct["warmup"] = traceback.format_exc(limit=3)
    try:
        direct["jit_launch"] = bench(lambda: flat_masked[grid](x, n, BLOCK=4096))
    except Exception:
        direct["jit_launch"] = traceback.format_exc(limit=2)
    info["direct_launch"] = direct

    # ---- (b) device scaling with program count -----------------------
    scaling = {}
    for cname, dtype, size in (("f32_16M", torch.float32, 4096 * 4096),
                               ("f16_16M", torch.float16, 4096 * 4096),
                               ("f32_1M", torch.float32, 1048576)):
        y = torch.randn(size, dtype=torch.float32, device=DEV).to(dtype)
        entry = {}
        for block in (4096, 8192, 16384):
            g = ((size + block - 1) // block,)
            try:
                entry["tile%d_progs_%d" % (block, g[0])] = bench(
                    lambda y=y, g=g, block=block, size=size:
                    flat_masked[g](y, size, BLOCK=block))
            except Exception:
                entry["tile%d_progs_%d" % (block, g[0])] = traceback.format_exc(limit=2)
        for chunk in (4096, 8192):
            for nprog in (20, 40, 80, 160, 320, 640):
                if nprog * chunk >= size * 4:
                    continue
                g = (nprog,)
                try:
                    entry["loop_chunk%d_progs_%d" % (chunk, nprog)] = bench(
                        lambda y=y, g=g, chunk=chunk, size=size:
                        loop_kernel[g](y, size, CHUNK=chunk, BLOCK=chunk))
                except Exception:
                    entry["loop_chunk%d_progs_%d" % (chunk, nprog)] = \
                        traceback.format_exc(limit=2)
        scaling[cname] = entry
    info["scaling"] = scaling

    with open(os.path.join(OUT, "probe2.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    print(json.dumps({k: info[k] for k in ("direct_launch", "scaling")}, indent=2)[:8000])


if __name__ == "__main__":
    main()
