"""Validate the direct CompiledKernel launch form (user args + trailing hidden None),
verify numerics, and sweep device configs on Ascend 910B."""
import json
import os
import time
import traceback

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = {"bench": {}, "notes": {}, "errors": []}


def save():
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "probe6.json"), "w") as f:
        json.dump(OUT, f, indent=1, default=str)


@triton.jit
def k_flat(X, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    y = tl.log(x.to(tl.float32)) * 0.4342944819032518
    tl.store(X + offs, y.to(X.dtype.element_ty), mask=mask)


@triton.jit
def k_nomask(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.log(x.to(tl.float32)) * 0.4342944819032518
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_nomask_l2(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.log2(x.to(tl.float32)) * 0.30102999566398120
    tl.store(X + offs, y.to(X.dtype.element_ty))


def host_time(fn, iters=300):
    fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    t2 = time.perf_counter()
    return (t1 - t0) / iters * 1e6, (t2 - t0) / iters * 1e6


def maxerr(x, fn):
    a = x.clone()
    fn(a)
    torch.npu.synchronize()
    return float((a.float() - torch.log10(x.float())).abs().max())


def record(name, fn, err=None):
    e = {}
    try:
        e["host_us"], e["amortized_us"] = host_time(fn)
    except Exception as exc:  # noqa: BLE001
        e["launch_err"] = f"{type(exc).__name__}: {str(exc)[:250]}"
    if err is not None:
        try:
            e["abs_err"] = err()
        except Exception as exc:  # noqa: BLE001
            e["err"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    OUT["bench"][name] = e
    save()
    print(name, json.dumps(e, default=str))


def main():
    from triton.runtime import driver
    import triton.backends.ascend.utils as autils

    gs = autils.get_backend_func("get_current_stream")
    dev_id = driver.active.get_current_device()
    OUT["notes"]["device"] = int(dev_id)

    dev = "npu:0"
    g = torch.Generator().manual_seed(0)
    ts = {}
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        for n in (1024, 65536, 1048576, 4096 * 4096):
            ts[(str(dt).split(".")[-1], n)] = torch.randn(n, generator=g).abs().to(dt).to(dev) + 0.05

    # ---- stream / metadata accessor costs ----
    record("cost::gs(dev)", lambda: gs(dev_id))
    record("cost::driver.get_current_stream(dev)", lambda: driver.active.get_current_stream(dev_id))
    record("cost::data_ptr", lambda: ts[("float32", 1024)].data_ptr())
    record("cost::numel+dtype", lambda: (ts[("float32", 1024)].numel(), ts[("float32", 1024)].dtype))

    x = ts[("float32", 4096 * 4096)]
    n = x.numel()
    blk = 32768
    ck = k_nomask.warmup(x, BLOCK=blk, grid=(1,))
    runner = ck[(n // blk, 1, 1)]
    st = gs(dev_id)

    # ---- direct launch forms (user args then trailing hidden slot) ----
    record("direct::streamless::fp32::16777216", lambda: runner(x, None),
           lambda: maxerr(x, lambda a: runner(a, None)))
    record("direct::stream_cached::fp32::16777216", lambda: runner(x, None, stream=st),
           lambda: maxerr(x, lambda a: runner(a, None, stream=st)))
    record("direct::stream_live::fp32::16777216", lambda: runner(x, None, stream=gs(dev_id)),
           lambda: maxerr(x, lambda a: runner(a, None, stream=gs(dev_id))))

    CACHE = {("float32", n): runner}

    def wrapped(A):
        r = CACHE.get((A.dtype, A.numel()))
        r(A, None, stream=gs(dev_id))

    record("wrapper::fp32::16777216", lambda: wrapped(x), lambda: maxerr(x, wrapped))
    record("baseline_jit::fp32::16777216", lambda: k_nomask[(n // blk,)](x, BLOCK=blk))

    # ---- device sweep with the cheap host path (cached-stream direct) ----
    def mk(jitfn, nn, b, nargs):
        t = ts[(dtype_of(jitfn, ts), nn)]
        return None

    def direct(jitfn, t, b, grid_n, stream=st):
        c = jitfn.warmup(t, *(() if jitfn is not k_flat else (t.numel(),)), BLOCK=b, grid=(1,))
        r = c[(grid_n, 1, 1)]
        if jitfn is k_flat:
            return lambda tt: r(tt, tt.numel(), None, stream=stream)
        return lambda tt: r(tt, None, stream=stream)

    for dt in ("float16", "float32", "bfloat16"):
        for nn in (1048576, 4096 * 4096):
            t = ts[(dt, nn)]
            for b in (8192, 16384, 32768):
                if nn % b:
                    continue
                f = direct(k_nomask, t, b, nn // b)
                record(f"dev_nomask{b}::{dt}::{nn}", lambda f=f, t=t: f(t),
                       lambda f=f, t=t: maxerr(t, f))
        t = ts[(dt, 4096 * 4096)]
        f = direct(k_nomask_l2, t, 32768, (4096 * 4096) // 32768)
        record(f"dev_log2_32768::{dt}::16777216", lambda f=f, t=t: f(t),
               lambda f=f, t=t: maxerr(t, f))

    # small / odd shapes through the masked kernel
    for dt in ("float16", "float32"):
        for nn in (1024, 65536):
            t = ts[(dt, nn)]
            b = 1024 if nn == 1024 else 16384
            f = direct(k_flat, t, b, triton.cdiv(nn, b))
            record(f"dev_flat{b}::{dt}::{nn}", lambda f=f, t=t: f(t),
                   lambda f=f, t=t: maxerr(t, f))

    # ---- non-divisible tail check (masked kernel, n = 65536 + 1) ----
    t = torch.randn(65537, generator=g).abs().to("npu:0") + 0.05
    f = direct(k_flat, t, 16384, triton.cdiv(65537, 16384))
    record("dev_flat16384::float32::65537", lambda f=f, t=t: f(t),
           lambda f=f, t=t: maxerr(t, f))
    OUT["notes"]["tail_ok"] = OUT["bench"]["dev_flat16384::float32::65537"].get("abs_err")

    # ---- build cost per new specialization ----
    t = ts[("float16", 4096 * 4096)]
    t0 = time.perf_counter()
    _ = k_nomask.warmup(t, BLOCK=8192, grid=(1,))[(2048, 1, 1)]
    OUT["notes"]["warmup_8192_fp16_s"] = time.perf_counter() - t0
    save()


def dtype_of(jitfn, ts):
    return "float32"


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        OUT["errors"].append(traceback.format_exc()[-3000:])
    save()
    print("ERRORS", json.dumps(OUT["errors"], indent=1)[:3000])
