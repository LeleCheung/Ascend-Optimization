"""Bounded target probe for xlogy_scalar_tensor.

Measures, on the real target:
  1. environment facts (versions, raw-stream fast path, device properties),
  2. host-side cost of each piece of the candidate call path (allocation,
     plain JIT dispatch, raw compiled-launcher dispatch, torch.xlogy),
  3. device-side throughput of the elementwise log kernel for several tile
     sizes and three math lowerings (tl.log, tl.log2*ln2, manual poly),
  4. numerical agreement of every math variant against the NPU oracle.

Writes a compact JSON artifact; prints the same dict.
"""
import json
import os
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

OUT = {}
MS = 1e-3


def bench_loop(fn, warmup=10, iters=50):
    """Amortized walltime per call (one sync at the end)."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters / MS


def bench_sync(fn, warmup=5, iters=20):
    """Per-call walltime with a synchronize after every call."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
        torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters / MS


# ---------------------------------------------------------------- environment
OUT["torch"] = torch.__version__
OUT["triton"] = triton.__version__
OUT["torch_npu"] = getattr(torch_npu, "__version__", "?")
raw = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
OUT["has_raw_stream"] = raw is not None
try:
    from triton.runtime import driver as _drv
    props = _drv.active.utils.get_device_properties(0)
    OUT["device_props"] = {k: int(v) for k, v in dict(props).items()
                           if isinstance(v, (int, bool))}
except Exception as exc:  # pragma: no cover
    OUT["device_props"] = "ERR:%s" % exc
if raw is not None:
    try:
        OUT["raw_stream_ok"] = int(raw(0)) == int(torch.npu.current_stream(0).npu_stream)
    except Exception as exc:
        OUT["raw_stream_ok"] = "ERR:%s" % exc


def stream_handle():
    if raw is not None:
        return raw(0)
    return torch.npu.current_stream(0).npu_stream


# ---------------------------------------------------------------- kernels
@triton.jit
def k_log(Y, OUTP, n, s, BLOCK: tl.constexpr, MASKED: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if MASKED:
        m = offs < n
        y = tl.load(Y + offs, mask=m, other=1.0).to(tl.float32)
        tl.store(OUTP + offs, (s * tl.log(y)).to(OUTP.dtype.element_ty), mask=m)
    else:
        y = tl.load(Y + offs).to(tl.float32)
        tl.store(OUTP + offs, (s * tl.log(y)).to(OUTP.dtype.element_ty))


@triton.jit
def k_log2(Y, OUTP, n, s, BLOCK: tl.constexpr, MASKED: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if MASKED:
        m = offs < n
        y = tl.load(Y + offs, mask=m, other=1.0).to(tl.float32)
        v = s * (tl.log2(y) * 0.6931471805599453)
        tl.store(OUTP + offs, v.to(OUTP.dtype.element_ty), mask=m)
    else:
        y = tl.load(Y + offs).to(tl.float32)
        v = s * (tl.log2(y) * 0.6931471805599453)
        tl.store(OUTP + offs, v.to(OUTP.dtype.element_ty))


@triton.jit
def k_poly(Y, OUTP, n, s, BLOCK: tl.constexpr, MASKED: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if MASKED:
        m = offs < n
        y = tl.load(Y + offs, mask=m, other=1.0).to(tl.float32)
    else:
        y = tl.load(Y + offs).to(tl.float32)
    # split mantissa/exponent, log(m * 2^e) = log(m) + e*ln2
    bits = y.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) - 127
    mm = (bits & 0x007FFFFF) | 0x3F800000
    mant = mm.to(tl.float32, bitcast=True)          # [1, 2)
    small = mant > 1.4142135623730951
    mant = tl.where(small, mant * 0.5, mant)
    e = tl.where(small, e + 1, e).to(tl.float32)
    t = (mant - 1.0) / (mant + 1.0)                 # |t| <= 0.1716
    t2 = t * t
    p = t * (2.0 + t2 * (0.6666666666666666 + t2 * (0.4 + t2 * (0.2857142857142857
        + t2 * 0.2222222222222222))))
    v = s * (p + e * 0.6931471805599453)
    v = tl.where(y != y, float("nan"), v)
    if MASKED:
        tl.store(OUTP + offs, v.to(OUTP.dtype.element_ty), mask=m)
    else:
        tl.store(OUTP + offs, v.to(OUTP.dtype.element_ty))


KERNELS = {"log": k_log, "log2": k_log2, "poly": k_poly}


def compiled(jitfn, y, out, n, s, block, masked):
    ck = jitfn.warmup(y, out, n, s, BLOCK=block, MASKED=masked, grid=(1,))
    return ck


# ---------------------------------------------------------------- op path costs
n0 = 1024
y0 = (torch.rand(n0, device="npu") + 0.05).to(torch.float16)
o0 = torch.empty_like(y0)
grid0 = ((n0 + 1023) // 1024,)

OUT["cost_py_noop_ms"] = bench_loop(lambda: None)
OUT["cost_alloc_ms"] = bench_loop(lambda: torch.empty(y0.shape, dtype=y0.dtype, device=y0.device))
OUT["cost_alloc_like_ms"] = bench_loop(lambda: torch.empty_like(y0))
OUT["cost_ptr_ms"] = bench_loop(lambda: y0.data_ptr())
OUT["cost_stream_ms"] = bench_loop(stream_handle)
OUT["cost_curstream_obj_ms"] = bench_loop(lambda: torch.npu.current_stream(0))
OUT["cost_ref_xlogy_ms"] = bench_loop(lambda: torch.xlogy(2.0, y0))
OUT["cost_ref_xlogy_sync_ms"] = bench_sync(lambda: torch.xlogy(2.0, y0))


def plain_call():
    k_log[grid0](y0, o0, n0, 2.0, BLOCK=1024, MASKED=True)


OUT["cost_plain_launch_ms"] = bench_loop(plain_call)
OUT["cost_plain_launch_sync_ms"] = bench_sync(plain_call)

ck0 = compiled(k_log, y0, o0, n0, 2.0, 1024, True)
launcher = ck0.run
OUT["launcher_has_launch"] = hasattr(launcher, "launch")
OUT["launcher_compile_only"] = bool(getattr(launcher, "compile_only", 0))
OUT["launcher_msprof_reg"] = bool(getattr(launcher, "enable_msprof_register_tensor", 0))
OUT["metadata_debug"] = bool(getattr(launcher.metadata, "debug_enabled", 0)
                             or getattr(launcher.metadata, "debug", 0))
if hasattr(launcher, "launch"):
    fn0 = ck0.function
    pm0 = ck0.packed_metadata

    def raw_call():
        launcher.launch(grid0, 1, 1, stream_handle(), fn0, pm0,
                        None, None, None, y0.data_ptr(), o0.data_ptr(), n0, 2.0, 1024, 1)

    OUT["cost_raw_launch_ms"] = bench_loop(raw_call)
    OUT["cost_raw_launch_sync_ms"] = bench_sync(raw_call)
    ref = torch.xlogy(2.0, y0)
    o0.zero_()
    raw_call()
    torch.npu.synchronize()
    OUT["raw_launch_matches"] = bool(torch.equal(o0, ref))

# ---------------------------------------------------------------- device throughput
CASES = []
for dtype, n, blocks in (
        (torch.float16, 4096 * 4096, (2048, 4096, 8192, 16384, 32768)),
        (torch.float32, 4096 * 4096, (1024, 2048, 4096, 8192, 16384)),
        (torch.float16, 1048576, (4096, 16384, 32768)),
        (torch.float32, 65536, (2048, 8192)),
):
    CASES.append((dtype, n, blocks))

dev = {}
for dtype, n, blocks in CASES:
    yy = (torch.rand(n, device="npu") + 0.05).to(dtype)
    oo = torch.empty_like(yy)
    ref = torch.xlogy(2.0, yy)
    tag = "%s_%d" % (str(dtype).split(".")[-1], n)
    entry = {"ref_ms": bench_loop(lambda: torch.xlogy(2.0, yy))}
    for b in blocks:
        masked = bool(n % b)
        for kname, jitfn in KERNELS.items():
            key = "%s_b%d" % (kname, b)
            if masked:
                key += "m"
            try:
                ck = compiled(jitfn, yy, oo, n, 2.0, b, masked)
                la = ck.run
                fn_c, pm_c = ck.function, ck.packed_metadata
                g = (triton.cdiv(n, b),)
                tail = (n, 2.0, b, 1) if masked else (n, 2.0, b, 0)

                def call(la=la, g=g, fn_c=fn_c, pm_c=pm_c, tail=tail):
                    la.launch(g, 1, 1, stream_handle(), fn_c, pm_c,
                              None, None, None, yy.data_ptr(), oo.data_ptr(), *tail)

                oo.zero_()
                call()
                torch.npu.synchronize()
                with torch.no_grad():
                    err = (oo.float() - ref.float()).abs()
                    nan_ok = bool(torch.equal(torch.isnan(oo), torch.isnan(ref)))
                entry[key] = bench_loop(call, warmup=5, iters=30)
                entry[key + "_err"] = float(err.nan_to_num(1e9).max().item())
                entry[key + "_nanok"] = nan_ok
            except Exception as exc:
                entry[key] = "ERR:%s" % str(exc)[:160]
    dev[tag] = entry

OUT["device"] = dev

art = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
try:
    with open(os.path.join(art, "probe.json"), "w") as fh:
        json.dump(OUT, fh, indent=2, default=str)
except Exception as exc:
    print("artifact write failed: %s" % exc)
print(json.dumps(OUT, indent=1, default=str))
