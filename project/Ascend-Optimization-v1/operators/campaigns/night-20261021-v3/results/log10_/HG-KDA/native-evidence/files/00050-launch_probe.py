"""Diagnose log10_ launch/host overhead and tile-size scaling on the target."""
import json
import os
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

OUT = {}


def props():
    p = {}
    try:
        from triton.runtime import driver

        dev = torch.npu.current_device()
        d = driver.active.utils.get_device_properties(dev)
        p["triton_driver_props"] = {k: int(v) for k, v in d.items()} if isinstance(d, dict) else str(d)
    except Exception as e:  # noqa: BLE001
        p["triton_driver_props"] = f"ERR {type(e).__name__}: {e}"
    try:
        tp = torch.npu.get_device_properties(0)
        p["torch_props"] = str(tp)
    except Exception as e:  # noqa: BLE001
        p["torch_props"] = f"ERR {type(e).__name__}: {e}"
    return p


@triton.jit
def k_flat(X, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    y = tl.log(x.to(tl.float32)) * 0.4342944819032518
    tl.store(X + offs, y.to(X.dtype.element_ty), mask=mask)


@triton.jit
def k_flat_f16(X, N, BLOCK: tl.constexpr):
    """Compute the transcendental in the input dtype (fp16 only)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    y = tl.log2(x) * 0.30102999566398120
    tl.store(X + offs, y.to(X.dtype.element_ty), mask=mask)


@triton.jit
def k_persist(X, N, NITER, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for i in range(0, NITER):
        offs = (pid + i * nprog) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        y = tl.log(x.to(tl.float32)) * 0.4342944819032518
        tl.store(X + offs, y.to(X.dtype.element_ty), mask=mask)


def bench_host_only(fn, iters=200):
    fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()  # host-only (async queue)
    torch.npu.synchronize()
    t2 = time.perf_counter()
    return {"host_us": (t1 - t0) / iters * 1e6, "h2d_us": (t2 - t0) / iters * 1e6}


def bench_iter_sync(fn, iters=100):
    fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
        torch.npu.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e6


def make_cases():
    dev = "npu:0"
    g = torch.Generator().manual_seed(0)
    cases = {}
    for dt in ("float16", "bfloat16", "float32"):
        for n in (1024, 1048576, 4096 * 4096):
            x = torch.randn(n, generator=g).abs().to(getattr(torch, dt)).to(dev) + 0.05
            cases[(dt, n)] = x
    return cases


def ref_err(x, fn):
    """max abs error of fn vs float64 torch reference on a copy."""
    a = x.clone()
    fn(a)
    torch.npu.synchronize()
    ref = torch.log10(x.double())
    got = a.double()
    return float((got - ref).abs().max()), float(((got - ref).abs() / ref.abs().clamp_min(1e-30)).max())


def main():
    OUT["props"] = props()
    cases = make_cases()
    dev = "npu:0"
    bench = {}

    def record(name, fn, err_fn=None):
        e = {}
        e.update(bench_host_only(fn))
        e["iter_sync_us"] = bench_iter_sync(fn)
        if err_fn is not None:
            try:
                e["abs_err"], e["rel_err"] = err_fn()
            except Exception as exc:  # noqa: BLE001
                e["err"] = f"ERR {type(exc).__name__}: {exc}"
        bench[name] = e

    # reference torch in-place
    for key, x in cases.items():
        record(
            f"torch::{key[0]}::{key[1]}",
            (lambda t=x: torch.log10_(t)),
            (lambda t=x: (float((torch.log10_(t.clone()).double() - torch.log10(t.double())).abs().max()), 0.0)),
        )

    # flat BLOCK sweep
    for blk in (1024, 2048, 4096, 8192, 16384, 32768, 65536):
        for key, x in cases.items():
            if key[1] < blk:
                continue
            grid = (triton.cdiv(key[1], blk),)
            record(
                f"flat{blk}::{key[0]}::{key[1]}",
                (lambda t=x, g=grid, b=blk: k_flat[g](t, t.numel(), BLOCK=b)),
            )

    # fp16-native compute path
    for blk in (8192, 32768):
        for key, x in cases.items():
            if key[0] != "float16" or key[1] < blk:
                continue
            grid = (triton.cdiv(key[1], blk),)
            record(
                f"flat_f16_{blk}::{key[0]}::{key[1]}",
                (lambda t=x, g=grid, b=blk: k_flat_f16[g](t, t.numel(), BLOCK=b)),
                (lambda t=x, g=grid, b=blk: ref_err(t, lambda a: k_flat_f16[g](a, a.numel(), BLOCK=b))),
            )

    # persistent grid = num_vectorcore * mult
    nvc = None
    try:
        from triton.runtime import driver

        d = driver.active.utils.get_device_properties(torch.npu.current_device())
        nvc = int(d.get("num_vectorcore") or d.get("num_aicore") or 0)
    except Exception:  # noqa: BLE001
        nvc = 0
    OUT["nvc"] = nvc
    for mult in (1, 2, 4):
        for blk in (8192, 32768):
            for key, x in cases.items():
                n = key[1]
                g = max(1, min(nvc * mult, triton.cdiv(n, blk)))
                niter = triton.cdiv(triton.cdiv(n, blk), g)
                if niter * g * blk < n:
                    continue
                record(
                    f"persist_m{mult}_{blk}::{key[0]}::{key[1]}",
                    (lambda t=x, gg=(g,), b=blk, it=niter: k_persist[gg](t, t.numel(), it, BLOCK=b)),
                    (lambda t=x, gg=(g,), b=blk, it=niter: ref_err(t, lambda a: k_persist[gg](a, a.numel(), it, BLOCK=b))),
                )

    OUT["bench"] = bench
    art = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, "launch_probe.json"), "w") as f:
        json.dump(OUT, f, indent=1, default=str)
    print(json.dumps(OUT["props"], indent=1)[:2000])


if __name__ == "__main__":
    main()
