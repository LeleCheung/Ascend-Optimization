"""Bounded target probe for upsample_bilinear2d.

Diagnostic only.  Measures, on the real Ascend target:
  * device properties (vector core count, UB size)
  * host-side launch cost of a Triton kernel vs the PyTorch reference
  * device cost of store-form / mask / tile-size variants of the 2x fast path
Nothing here is a candidate submission.
"""

import json
import os
import time
import traceback

import torch
import triton
import triton.language as tl

RES = {}


def rec(key, val):
    RES[key] = val
    print("%s = %s" % (key, val), flush=True)


def get_artifacts_dir():
    d = os.environ.get("KGS_DEBUG_ARTIFACTS", "")
    if d and os.path.isdir(d):
        return d
    return os.getcwd()


# ---------------------------------------------------------------- env facts
try:
    import torch_npu  # noqa: F401

    rec("torch_npu", getattr(torch_npu, "__version__", "?"))
except Exception as exc:  # pragma: no cover
    rec("torch_npu_import_error", repr(exc))

rec("torch", torch.__version__)
rec("triton", triton.__version__)

DEV = None
try:
    DEV = torch.npu.current_device()
    rec("current_device", DEV)
except Exception as exc:
    rec("current_device_error", repr(exc))

try:
    from triton.runtime import driver as _drv

    _props = _drv.active.utils.get_device_properties(DEV)
    rec("triton_props", dict(_props) if isinstance(_props, dict) else str(_props))
except Exception as exc:
    rec("triton_props_error", repr(exc))

try:
    rec("torch_npu_props", str(torch.npu.get_device_properties(DEV)))
except Exception as exc:
    rec("torch_npu_props_error", repr(exc))

try:
    rec("ub_size", triton.runtime.driver.active.utils.get_device_properties(DEV).get("ub_size"))
except Exception as exc:
    rec("ub_size_error", repr(exc))


# ---------------------------------------------------------------- kernels
@triton.jit
def _touch(ptr):
    tl.store(ptr + tl.program_id(0), 1.0)


@triton.jit
def _fast(
    ptr_o,
    ptr_i,
    s_n,
    s_c,
    s_h,
    rh,
    C: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    BR: tl.constexpr,
    IWP: tl.constexpr,
    CM: tl.constexpr,
    STORE2D: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    oh = rows % OH
    nc = rows // OH
    n = nc // C
    c = nc - n * C

    real_h = tl.maximum((oh.to(tl.float32) + 0.5) * rh - 0.5, 0.0)
    h0f = real_h.to(tl.int32)
    hl = real_h - h0f.to(tl.float32)
    h0 = tl.minimum(h0f, IH - 1)
    h1 = tl.minimum(h0 + 1, IH - 1)

    base = n * s_n + c * s_c
    rr0 = (base + h0 * s_h)[:, None]
    rr1 = (base + h1 * s_h)[:, None]

    k = tl.arange(0, IWP)
    ka = k[None, :]
    hl2 = hl[:, None]
    h0l2 = 1.0 - hl2

    if CM:
        lm = tl.broadcast_to((ka < IW) & (rows[:, None] < OH * C), (BR, IWP))
        v = h0l2 * tl.load(ptr_i + (rr0 + ka), mask=lm, other=0.0).to(tl.float32) \
            + hl2 * tl.load(ptr_i + (rr1 + ka), mask=lm, other=0.0).to(tl.float32)
    else:
        v = h0l2 * tl.load(ptr_i + (rr0 + ka)).to(tl.float32) \
            + hl2 * tl.load(ptr_i + (rr1 + ka)).to(tl.float32)

    il = tl.broadcast_to(tl.maximum(k - 1, 0)[None, :], (BR, IWP))
    ir = tl.broadcast_to(tl.minimum(k + 1, IW - 1)[None, :], (BR, IWP))
    vm = tl.gather(v, il, 1)
    vp = tl.gather(v, ir, 1)

    even = 0.25 * vm + 0.75 * v
    odd = 0.75 * v + 0.25 * vp

    if STORE2D:
        val = tl.reshape(tl.join(even, odd), (BR, 2 * IWP))
        ooff = (rows * OW)[:, None] + tl.arange(0, 2 * IWP)[None, :]
        if CM:
            om = tl.broadcast_to(
                (tl.arange(0, 2 * IWP) < OW)[None, :]
                & (rows[:, None] < OH * C), (BR, 2 * IWP))
            tl.store(ptr_o + ooff, val.to(ptr_o.dtype.element_ty), mask=om)
        else:
            tl.store(ptr_o + ooff, val.to(ptr_o.dtype.element_ty))
    else:
        val = tl.join(even, odd)
        j = tl.arange(0, 2)
        ooff = (rows * OW)[:, None, None] + (ka * 2)[:, :, None] + j[None, None, :]
        if CM:
            om = tl.broadcast_to(
                (ka < IW)[:, :, None] & (rows[:, None, None] < OH * C), (BR, IWP, 2))
            tl.store(ptr_o + ooff, val.to(ptr_o.dtype.element_ty), mask=om)
        else:
            tl.store(ptr_o + ooff, val.to(ptr_o.dtype.element_ty))


@triton.jit
def _floor(ptr_o, ptr_i, s_h, IH: tl.constexpr, IWP: tl.constexpr, BR: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rr = ((rows % IH) * s_h)[:, None]
    v = tl.load(ptr_i + (rr + tl.arange(0, IWP)[None, :])).to(tl.float32)
    val = tl.interleave(v, v)
    ooff = (rows * (2 * IWP))[:, None] + tl.arange(0, 2 * IWP)[None, :]
    tl.store(ptr_o + ooff, val.to(ptr_o.dtype.element_ty))


# ---------------------------------------------------------------- timing
def time_wall(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def time_host(fn, iters=200, warmup=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    return (t1 - t0) / iters * 1e6


def ref_op(x, oh, ow, ac):
    return torch.ops.aten.upsample_bilinear2d(x, [oh, ow], ac, None, None)


# ---------------------------------------------------------------- probes
DEVICE = "npu:0" if DEV is None else DEV


def probe_launch():
    x = torch.ones(64, device=DEVICE, dtype=torch.float32)
    try:
        _touch[(1,)](x)
        torch.npu.synchronize()
        rec("host_launch_triton_us", time_host(lambda: _touch[(1,)](x)))
    except Exception as exc:
        rec("host_launch_triton_error", repr(exc))

    try:
        rec("host_ref_small_us", time_host(lambda: ref_op(
            SMALL_I, SH, SW, False)))
    except Exception as exc:
        rec("host_ref_small_error", repr(exc))


def check(name, out, ref):
    try:
        d = (out.float() - ref.float()).abs().max().item()
        rec("maxdiff_" + name, d)
    except Exception as exc:
        rec("maxdiff_" + name + "_error", repr(exc))


SMALL = (1, 16, 64, 64)
SH, SW = 128, 128
BIG = (4, 32, 128, 128)
BH, BW = 256, 256
SMALL_I = torch.randn(SMALL, device=DEVICE, dtype=torch.float16)
BIG_I = torch.randn(BIG, device=DEVICE, dtype=torch.float16)


def run_variant(tag, x, OH, OW, BR, CM, STORE2D, iters=20):
    N, C, IH, IW = x.shape
    n_rows = N * C * OH
    IWP = triton.next_power_of_2(IW)
    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    grid = (n_rows // BR,)
    s_n, s_c, s_h, _ = x.stride()

    def call():
        _fast[grid](out, x, s_n, s_c, s_h, IH / OH, C=C, IH=IH, IW=IW, OH=OH,
                    OW=OW, BR=BR, IWP=IWP, CM=CM, STORE2D=STORE2D)

    call()
    torch.npu.synchronize()
    rec("wall_" + tag, time_wall(call, iters=iters))
    rec("host_" + tag, time_host(call, iters=100))
    ref = ref_op(x, OH, OW, False)
    check(tag, out, ref)
    return out


def probe_kernels():
    # small geometry
    for tag, br, cm, s2 in [
        ("s_seed_br16", 16, True, False),
        ("s_nomask3d_br16", 16, False, False),
        ("s_nomask2d_br16", 16, False, True),
        ("s_nomask2d_br64", 64, False, True),
    ]:
        try:
            run_variant(tag, SMALL_I, SH, SW, br, cm, s2, iters=50)
        except Exception as exc:
            rec("err_" + tag, repr(exc))

    # big geometry
    for tag, br, cm, s2 in [
        ("b_seed_br16", 16, True, False),
        ("b_nomask3d_br16", 16, False, False),
        ("b_nomask2d_br16", 16, False, True),
        ("b_nomask2d_br32", 32, False, True),
        ("b_nomask2d_br64", 64, False, True),
        ("b_nomask2d_br128", 128, False, True),
    ]:
        try:
            run_variant(tag, BIG_I, BH, BW, br, cm, s2, iters=10)
        except Exception as exc:
            rec("err_" + tag, repr(exc))

    # pure I/O floor at the big geometry (read once, write 4x volume)
    N, C, IH, IW = BIG
    n_rows = N * C * BH
    out = torch.empty((N, C, BH, BW), device=DEVICE, dtype=torch.float16)
    for br in (32, 64):
        IWP = triton.next_power_of_2(IW)
        grid = (n_rows // br,)

        def call(br=br, grid=grid, IWP=IWP):
            _floor[grid](out, BIG_I, IW, IH=IH, IWP=IWP, BR=br)

        try:
            call()
            torch.npu.synchronize()
            rec("wall_floor_br%d" % br, time_wall(call, iters=10))
        except Exception as exc:
            rec("err_floor_br%d" % br, repr(exc))


def probe_ref():
    for tag, x, oh, ow in (("small", SMALL_I, SH, SW), ("big", BIG_I, BH, BW)):
        try:
            rec("wall_ref_" + tag, time_wall(lambda x=x, o=oh, w=ow: ref_op(x, o, w, False), iters=20))
        except Exception as exc:
            rec("wall_ref_" + tag + "_error", repr(exc))


def main():
    probe_launch()
    probe_ref()
    probe_kernels()


try:
    main()
except Exception:
    rec("fatal", traceback.format_exc())

path = os.path.join(get_artifacts_dir(), "probe_result.json")
with open(path, "w") as fh:
    json.dump(RES, fh, indent=2, default=str)
print("wrote", path, flush=True)
