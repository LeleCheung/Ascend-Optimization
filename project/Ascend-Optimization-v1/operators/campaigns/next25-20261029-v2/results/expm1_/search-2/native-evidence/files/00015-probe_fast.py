"""Probe 3 (final diagnostic): (a) can a pre-resolved CompiledKernel be launched
directly, and how much does that cut the 62 us JIT launch cost? (b) with a cheap
host path, what is the real device time for 16M elements as a function of tile
and program count, and of the math variant? (c) accuracy of the variants under
the harness tolerance model.

Writes probe3.json into $KGS_DEBUG_ARTIFACTS.
"""

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
from triton.runtime import driver as _drv

OUT = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
DEV = "npu:0"
ITERS = 30
LOG2E = 1.4426950408889634


@triton.jit
def flat_f32math(a_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(a_ptr + offs, mask=m, other=0.0)
    y = tl.exp(x.to(tl.float32)) - 1.0
    tl.store(a_ptr + offs, y.to(a_ptr.dtype.element_ty), mask=m)


@triton.jit
def flat_exp2(a_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(a_ptr + offs, mask=m, other=0.0)
    y = tl.exp2(x.to(tl.float32) * 1.4426950408889634) - 1.0
    tl.store(a_ptr + offs, y.to(a_ptr.dtype.element_ty), mask=m)


@triton.jit
def flat_native(a_ptr, n, BLOCK: tl.constexpr):
    """No fp32 upcast: run the transcendental in the buffer dtype."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(a_ptr + offs, mask=m, other=0.0)
    y = tl.exp(x) - 1.0
    tl.store(a_ptr + offs, y, mask=m)


@triton.jit
def loop_f32math(a_ptr, n, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
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
        "host_submit_us": (t1 - t0) / ITERS * 1e6,
        "pipeline_us": (t2 - t0) / ITERS * 1e6,
    }


def main():
    info = {}
    device = _drv.active.get_current_device()
    info["device_index"] = device
    try:
        info["props"] = _drv.active.utils.get_device_properties(device)
    except Exception:
        info["props"] = traceback.format_exc(limit=1)

    # ---------- (a) direct launch machinery ----------
    x = torch.randn(1048576, dtype=torch.float32, device=DEV)
    n = x.numel()
    ck = flat_f32math.warmup(x, n, BLOCK=16384, grid=((n + 16383) // 16384,))
    d = {"ck": str(type(ck))}
    try:
        d["run_sig"] = str([a for a in dir(ck) if a in ("run", "_run", "function",
                                                        "packed_metadata", "launch_metadata")])
        stream = _drv.active.get_current_stream(device)
        d["stream"] = repr(stream)[:80]
        d["function"] = repr(ck.function)[:80]
        d["pmd"] = repr(ck.packed_metadata)[:200]
    except Exception:
        d["meta"] = traceback.format_exc(limit=2)

    launcher = ck.run
    grid0 = (n + 16383) // 16384

    def call_tensor():
        st = _drv.active.get_current_stream(device)
        launcher(grid0, 1, 1, st, ck.function, ck.packed_metadata, None, None,
                 None, x, n, 16384)

    def call_intptr():
        st = _drv.active.get_current_stream(device)
        launcher(grid0, 1, 1, st, ck.function, ck.packed_metadata, None, None,
                 None, x.data_ptr(), n, 16384)

    def call_intptr_nostream():
        launcher(grid0, 1, 1, STREAM_CACHE[0], ck.function, ck.packed_metadata,
                 None, None, None, x.data_ptr(), n, 16384)

    STREAM_CACHE = [_drv.active.get_current_stream(device)]

    for name, fn in (("tensor_arg", call_tensor),
                     ("intptr_arg", call_intptr),
                     ("intptr_cached_stream", call_intptr_nostream),
                     ("jit_normal", lambda: flat_f32math[(grid0,)](x, n, BLOCK=16384))):
        try:
            d[name] = bench(fn)
        except Exception:
            d[name] = traceback.format_exc(limit=3)

    # verify the int-pointer launch actually computes
    try:
        v = torch.randn(4096, dtype=torch.float32, device=DEV)
        ref = torch.expm1(v)
        ck2 = flat_f32math.warmup(v, v.numel(), BLOCK=16384, grid=(1,))
        ck2.run(1, 1, 1, _drv.active.get_current_stream(device), ck2.function,
                ck2.packed_metadata, None, None, None, v.data_ptr(), v.numel(), 16384)
        torch.npu.synchronize()
        d["intptr_correct"] = bool(torch.allclose(v, ref, atol=1e-6, rtol=1e-6))
    except Exception:
        d["intptr_correct"] = traceback.format_exc(limit=3)
    info["direct"] = d

    # ---------- (b) device time with the cheap host path ----------
    def fast_call(kern, size, xin, block, nprog=None, chunk=None):
        if chunk is None:
            g = ((size + block - 1) // block,)
            ck_ = kern.warmup(xin, size, BLOCK=block, grid=g)
            args = (xin.data_ptr(), size, block)
        else:
            g = (nprog,)
            ck_ = kern.warmup(xin, size, CHUNK=chunk, BLOCK=chunk, grid=g)
            args = (xin.data_ptr(), size, chunk, chunk)
        run_ = ck_.run
        fn = ck_.function
        pmd = ck_.packed_metadata
        g0 = g[0]

        def call():
            run_(g0, 1, 1, STREAM_CACHE[0], fn, pmd, None, None, None, *args)

        return call

    scaling = {}
    for cname, dtype, size in (("f32_16M", torch.float32, 4096 * 4096),
                               ("f16_16M", torch.float16, 4096 * 4096),
                               ("f32_1M", torch.float32, 1048576),
                               ("f16_1024", torch.float16, 1024)):
        y = torch.randn(size, dtype=torch.float32, device=DEV).to(dtype)
        entry = {}
        for kern_name, kern in (("f32math", flat_f32math), ("exp2", flat_exp2),
                                ("native", flat_native)):
            for block in (16384, 32768):
                try:
                    entry["%s_flat%d" % (kern_name, block)] = bench(
                        fast_call(kern, size, y, block))
                except Exception:
                    entry["%s_flat%d" % (kern_name, block)] = traceback.format_exc(limit=2)
        for chunk in (8192, 16384):
            for nprog in (40, 80, 160, 320):
                try:
                    entry["f32math_loop%d_x%d" % (chunk, nprog)] = bench(
                        fast_call(loop_f32math, size, y, chunk, nprog=nprog, chunk=chunk))
                except Exception:
                    entry["f32math_loop%d_x%d" % (chunk, nprog)] = traceback.format_exc(limit=2)
        scaling[cname] = entry
    info["scaling"] = scaling

    # ---------- (c) accuracy of the math variants ----------
    acc = {}
    try:
        nchk = 1 << 20
        g = torch.Generator().manual_seed(1729)
        base = torch.randn(nchk, generator=g)
        for dt, atol in ((torch.float16, 1.5e-2), (torch.bfloat16, 1.5e-2),
                         (torch.float32, 1e-4)):
            src = base.to(dt).to(DEV)
            src[:7] = torch.tensor([0.0, -0.0, float("nan"), float("inf"),
                                    -float("inf"), -1.0, 1.0], dtype=dt, device=DEV)
            ref = torch.expm1(src)
            per = {}
            for kern_name, kern in (("f32math", flat_f32math), ("exp2", flat_exp2),
                                    ("native", flat_native)):
                cand = src.clone()
                kern[(1,)](cand, cand.numel(), BLOCK=32768)
                torch.npu.synchronize()
                a = cand.to(torch.float32)
                b = ref.to(torch.float32)
                nan_ok = torch.equal(torch.isnan(a), torch.isnan(b))
                diff = (a - b).abs()
                lim = atol + atol * b.abs()
                bad = int((diff > lim).sum().item())
                per[kern_name] = {
                    "max_abs_diff": float(diff.nan_to_num(nan=0.0).max()),
                    "violations": bad,
                    "nan_pattern_equal": bool(nan_ok),
                }
            acc[str(dt)] = per
    except Exception:
        acc["error"] = traceback.format_exc(limit=3)
    info["accuracy"] = acc

    with open(os.path.join(OUT, "probe3.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    print(json.dumps(info, indent=2)[:12000])


if __name__ == "__main__":
    main()
