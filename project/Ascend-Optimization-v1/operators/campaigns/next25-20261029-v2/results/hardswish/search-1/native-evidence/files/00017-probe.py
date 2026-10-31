"""Bounded target-side diagnosis for hardswish.

(1) Host-path decomposition at n=1024 f16 (per-call microseconds).
(2) Launch-geometry sweep for 4096x4096 f16 and f32 (device-side dominated).
(3) Confirm whether the seed's cached low-level launch path is actually used.

Writes a compact JSON artifact; no source under test is modified.
"""
import json
import os
import sys
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = {"env": {}, "host": {}, "big": {}, "seed_state": {}}


def bench(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


# ---------------------------------------------------------------- kernels
@triton.jit
def k_loop(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for start in range(tl.program_id(0) * BLOCK, N, step):
        offset = start + lane
        x = tl.load(X + offset, offset < N, other=0).to(tl.float32)
        inner = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
        tl.store(Y + offset, x * inner * (1.0 / 6.0), offset < N)


@triton.jit
def k_flat(X, Y, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs).to(tl.float32)
    inner = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
    tl.store(Y + offs, x * inner * (1.0 / 6.0))


@triton.jit
def k_flat_masked(X, Y, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(X + offs, mask=m, other=0.0).to(tl.float32)
    inner = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0)
    tl.store(Y + offs, x * inner * (1.0 / 6.0), mask=m)


@triton.jit
def k_noop(X):
    pass


def main():
    dev = torch.npu.current_device()
    OUT["env"] = {
        "device": str(torch.npu.get_device_name(dev)),
        "triton": triton.__version__,
        "num_vectorcore": None,
        "num_aicore": None,
    }
    try:
        from triton.runtime import driver

        props = driver.active.utils.get_device_properties(dev)
        OUT["env"]["props_keys"] = sorted(str(k) for k in props.keys())
        OUT["env"]["num_vectorcore"] = props.get("num_vectorcore")
        OUT["env"]["num_aicore"] = props.get("num_aicore")
    except Exception as exc:  # pragma: no cover
        OUT["env"]["props_error"] = repr(exc)

    # -------------------------------------------------- host decomposition
    x = torch.randn(1024, device="npu", dtype=torch.float16)
    out = torch.empty_like(x)
    ref = torch.nn.functional.hardswish(x)
    n = x.numel()

    raw = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)

    # compile a one-shot specialization for the tiny case, then time pieces
    k_flat.warmup(x, out, BLOCK=1024, grid=(1,))
    tiny_launch = k_loop.warmup(x, out, n, 1024, grid=(1,))
    host = OUT["host"]
    host["loop_overhead_us"] = bench(lambda: None, 2000, 100)
    host["data_ptr_us"] = bench(lambda: x.data_ptr(), 2000, 100)
    host["numel_us"] = bench(lambda: x.numel(), 2000, 100)
    host["empty_like_us"] = bench(lambda: torch.empty_like(x), 2000, 100)
    host["contiguous_us"] = bench(lambda: x.contiguous(), 2000, 100)
    if raw is not None:
        host["raw_stream_us"] = bench(lambda: raw(dev), 2000, 100)
    host["current_stream_us"] = bench(
        lambda: torch.npu.current_stream(dev).npu_stream, 2000, 100)
    host["torch_ref_us"] = bench(lambda: torch.nn.functional.hardswish(x), 2000, 100)
    host["jit_flat_1024_us"] = bench(
        lambda: k_flat[(1,)](x, out, BLOCK=1024), 2000, 100)
    host["jit_loop_1024_us"] = bench(
        lambda: k_loop[(1,)](x, out, n, 1024), 2000, 100)
    host["jit_noop_us"] = bench(lambda: k_noop[(1,)](x), 2000, 100)

    # low-level launch with integer addresses (seed fast path shape)
    ck = tiny_launch
    launcher = ck.run
    OUT["seed_state"]["launcher_flags"] = {
        "compile_only": bool(getattr(launcher, "compile_only", False)),
        "msprof": bool(getattr(launcher, "enable_msprof_register_tensor", False)),
        "debug_enabled": bool(getattr(launcher.metadata, "debug_enabled", False)),
        "debug": bool(getattr(launcher.metadata, "debug", False)),
    }
    try:
        L, F, P = launcher.launch, ck.function, ck.packed_metadata
        xp, yp = x.data_ptr(), out.data_ptr()
        getter = (lambda: raw(dev)) if raw is not None else (
            lambda: torch.npu.current_stream(dev).npu_stream)
        host["lowlevel_launch_int_us"] = bench(
            lambda: L((1,), 1, 1, getter(), F, P, None, None, None, xp, yp, n, 1024),
            2000, 100)
        host["lowlevel_plus_alloc_us"] = bench(
            _mk_alloc_launch(L, F, P, getter, x, n, 1024), 2000, 100)
    except Exception as exc:
        host["lowlevel_error"] = repr(exc)

    # ------------------------------------------------------ seed behaviour
    try:
        sys.path.insert(0, os.getcwd())
        import tmp.main as seed

        seed.run(x)
        OUT["seed_state"]["entries"] = {
            str(k): ("None" if v is None else "fast") for k, v in seed._ENTRIES.items()
        }
        OUT["seed_state"]["run_us"] = bench(lambda: seed.run(x), 2000, 100)
        xs = torch.randn(7, 9, device="npu", dtype=torch.float16)
        base = torch.empty((7, 9, 2), device="npu", dtype=torch.float16)
        base[..., 0] = xs
        view = base[..., 0]
        got = seed.run(view)
        exp = torch.nn.functional.hardswish(view)
        OUT["seed_state"]["strided_max_abs_err"] = float(
            (got - exp).abs().max().item())
    except Exception as exc:
        OUT["seed_state"]["error"] = repr(exc)

    # ------------------------------------------------- geometry sweep, big
    def sweep(tag, dt):
        xx = torch.randn(4096, 4096, device="npu", dtype=dt)
        yy = torch.empty_like(xx)
        nb = xx.numel()
        expected = torch.nn.functional.hardswish(xx)
        res = {"torch_ref_us": bench(lambda: torch.nn.functional.hardswish(xx), 60, 20)}
        for blk in (4096, 8192, 16384):
            grid = nb // blk
            try:
                k_flat.warmup(xx, yy, BLOCK=blk, grid=(1,))
                yy.zero_()
                k_flat[(grid,)](xx, yy, BLOCK=blk)
                torch.npu.synchronize()
                err = float((yy - expected).abs().max().item())
                t = bench(lambda: k_flat[(grid,)](xx, yy, BLOCK=blk), 60, 20)
                res[f"flat_blk{blk}_grid{grid}_us"] = t
                res[f"flat_blk{blk}_err"] = err
            except Exception as exc:
                res[f"flat_blk{blk}_error"] = repr(exc)[:300]
        for g in (40, 80, 160, 320):
            try:
                k_loop.warmup(xx, yy, nb, 4096, grid=(1,))
                yy.zero_()
                k_loop[(g,)](xx, yy, nb, 4096)
                torch.npu.synchronize()
                err = float((yy - expected).abs().max().item())
                t = bench(lambda: k_loop[(g,)](xx, yy, nb, 4096), 60, 20)
                res[f"loop_blk4096_grid{g}_us"] = t
                res[f"loop_blk4096_grid{g}_err"] = err
            except Exception as exc:
                res[f"loop_blk4096_grid{g}_error"] = repr(exc)[:300]
        OUT["big"][tag] = res

    sweep("f16_4096x4096", torch.float16)
    sweep("f32_4096x4096", torch.float32)
    # medium case: does grid matter at 1M elements?
    xm = torch.randn(1048576, device="npu", dtype=torch.float16)
    ym = torch.empty_like(xm)
    OUT["big"]["f16_1M"] = {}
    for blk in (1024, 4096):
        grid = 1048576 // blk
        try:
            k_flat.warmup(xm, ym, BLOCK=blk, grid=(1,))
            OUT["big"]["f16_1M"][f"flat_blk{blk}_grid{grid}_us"] = bench(
                lambda: k_flat[(grid,)](xm, ym, BLOCK=blk), 200, 50)
        except Exception as exc:
            OUT["big"]["f16_1M"][f"flat_blk{blk}_error"] = repr(exc)[:300]
    OUT["big"]["f16_1M"]["loop_grid40_us"] = bench(
        lambda: k_loop[(40,)](xm, ym, 1048576, 4096), 200, 50)
    OUT["big"]["f16_1M"]["torch_ref_us"] = bench(
        lambda: torch.nn.functional.hardswish(xm), 200, 50)

    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "hardswish_probe.json"), "w") as fh:
        json.dump(OUT, fh, indent=2, sort_keys=True)
    print(json.dumps(OUT, indent=2, sort_keys=True))


def _mk_alloc_launch(L, F, P, getter, x, n, blk):
    def call():
        o = torch.empty_like(x)
        L((1,), 1, 1, getter(), F, P, None, None, None,
          x.data_ptr(), o.data_ptr(), n, blk)
        return o

    return call


if __name__ == "__main__":
    main()
