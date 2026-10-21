"""Measure cheap per-call launch forms (direct CompiledKernel runner) and re-tune
device configs now that the host floor can be lowered, on Ascend 910B."""
import functools
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
    with open(os.path.join(ART, "probe4.json"), "w") as f:
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


@triton.jit
def k_nomask_mulfirst(X, BLOCK: tl.constexpr):
    """Scale before the transcendental to keep one op on the fp32 value."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.log(x.to(tl.float32) * 1.0000000000000002) * 0.4342944819032518
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


def record(name, fn, err=None):
    e = {}
    try:
        e["host_us"], e["amortized_us"] = host_time(fn)
    except Exception as exc:  # noqa: BLE001
        e["launch_err"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    if err is not None:
        try:
            e["abs_err"] = err()
        except Exception as exc:  # noqa: BLE001
            e["err"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    OUT["bench"][name] = e
    save()
    print(name, json.dumps(e, default=str))


def maxerr(x, fn):
    a = x.clone()
    fn(a)
    torch.npu.synchronize()
    return float((a.float() - torch.log10(x.float())).abs().max())


def main():
    from triton.runtime import driver

    # ---- stream acquisition cost candidates ----
    cands = {
        "driver.get_current_stream(0)": lambda: driver.active.get_current_stream(0),
        "driver.get_current_stream(None)": lambda: driver.active.get_current_stream(None),
        "torch.npu.current_stream()": lambda: torch.npu.current_stream(),
        "torch.npu.current_stream().npu_stream": lambda: torch.npu.current_stream().npu_stream,
    }
    try:
        import triton.backends.ascend.utils as autils

        OUT["notes"]["get_backend_func_src"] = __import__("inspect").getsource(autils.get_backend_func)[:800]
        gs = autils.get_backend_func("get_current_stream")
        cands["cached backend func get_current_stream(0)"] = lambda: gs(0)
        cands["cached backend func get_current_stream(None)"] = lambda: gs(None)
    except Exception as exc:  # noqa: BLE001
        OUT["errors"].append(f"get_backend_func: {type(exc).__name__}: {exc}")
    try:
        OUT["notes"]["torch_C_stream_names"] = [a for a in dir(torch_npu._C) if "tream" in a or "urrent" in a][:40]
    except Exception as exc:  # noqa: BLE001
        OUT["errors"].append(f"torch_npu._C: {exc}")
    for k, fn in cands.items():
        try:
            v = fn()
            r = {"value": int(v) if isinstance(v, int) else str(v)}
            r["host_us"], _ = host_time(fn)
            OUT["bench"][f"stream::{k}"] = r
            print("stream", k, json.dumps(r, default=str))
        except Exception as exc:  # noqa: BLE001
            OUT["bench"][f"stream::{k}"] = {"err": f"{type(exc).__name__}: {str(exc)[:200]}"}
    save()

    # ---- tensors ----
    dev = "npu:0"
    g = torch.Generator().manual_seed(0)
    ts = {}
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        for n in (1024, 65536, 1048576, 4096 * 4096):
            ts[(str(dt).split(".")[-1], n)] = torch.randn(n, generator=g).abs().to(dt).to(dev) + 0.05

    x = ts[("float32", 4096 * 4096)]
    n = x.numel()
    blk = 32768
    grid = (n // blk, 1, 1)

    # ---- JIT launch (status quo) ----
    record("jit::k_nomask::float32::16777216", lambda: k_nomask[(n // blk,)](x, BLOCK=blk))

    # ---- direct CompiledKernel launch variants ----
    ck = k_nomask.warmup(x, BLOCK=blk, grid=(1,))
    st = driver.active.get_current_stream(0)
    dev_id = driver.active.get_current_device()
    OUT["notes"]["stream_val"] = int(st)
    OUT["notes"]["device_val"] = int(dev_id)

    r_none = ck[grid]
    record("direct_noargs::float32::16777216", lambda: r_none(x),
           lambda: maxerr(x, lambda a: r_none(a)))
    r_st = ck[grid]
    record("direct_stream::float32::16777216", lambda: r_st(x, stream=st),
           lambda: maxerr(x, lambda a: r_st(a, stream=st)))
    part = functools.partial(r_st, stream=st)
    record("direct_partial::float32::16777216", lambda: part(x),
           lambda: maxerr(x, lambda a: part(a)))
    st_live = lambda: driver.active.get_current_stream(0)
    record("direct_stream_lookup::float32::16777216", lambda: r_st(x, stream=st_live()))

    # wrapper-style: dict lookup + launch
    CACHE = {("float32", n): part}

    def wrapped(A):
        e = CACHE.get((A.dtype, A.numel()))
        if e is None:
            raise RuntimeError("miss")
        e(A)

    record("wrapper_lookup::float32::16777216", lambda: wrapped(x),
           lambda: maxerr(x, wrapped))

    # ---- device-side sweep with the cheap host path ----
    def direct(jitfn, blk_, gshape, extra_args=()):
        c = jitfn.warmup(gshape[0], *extra_args, BLOCK=blk_, grid=(1,))
        rr = c[(gshape[1], 1, 1)]
        return functools.partial(rr, stream=st)

    for dt in ("float16", "float32"):
        for nn in (1048576, 4096 * 4096):
            t = ts[(dt, nn)]
            for b in (8192, 16384, 32768):
                if nn % b:
                    continue
                f = direct(k_nomask, b, (t, nn // b))
                record(f"direct_nomask{b}::{dt}::{nn}", lambda f=f, t=t: f(t),
                       lambda f=f, t=t: maxerr(t, f))
    # log2 variant + persistent-ish wider tile on the largest case
    for dt in ("float16", "float32"):
        t = ts[(dt, 4096 * 4096)]
        f = direct(k_nomask_l2, 32768, (t, (4096 * 4096) // 32768))
        record(f"direct_log2_32768::{dt}::16777216", lambda f=f, t=t: f(t),
               lambda f=f, t=t: maxerr(t, f))
        f2 = direct(k_nomask_l2, 16384, (t, (4096 * 4096) // 16384))
        record(f"direct_log2_16384::{dt}::16777216", lambda f=f2, t=t: f2(t),
               lambda f=f2, t=t: maxerr(t, f2))
    for dt in ("bfloat16",):
        for nn in (1048576, 4096 * 4096):
            t = ts[(dt, nn)]
            f = direct(k_nomask, 32768, (t, nn // 32768))
            record(f"direct_nomask32768::{dt}::{nn}", lambda f=f, t=t: f(t),
                   lambda f=f, t=t: maxerr(t, f))

    # ---- masked fallback (odd shapes) with cheap host path ----
    t = ts[("float32", 65536)]
    f = direct(k_flat, 16384, (t, 4), (t.numel(),))
    record("direct_flat16384::float32::65536", lambda f=f, t=t: f(t, t.numel()),
           lambda f=f, t=t: maxerr(t, lambda a: f(a, a.numel())))
    t = ts[("float16", 1024)]
    f = direct(k_flat, 1024, (t, 1), (t.numel(),))
    record("direct_flat1024::float16::1024", lambda f=f, t=t: f(t, t.numel()),
           lambda f=f, t=t: maxerr(t, lambda a: f(a, a.numel())))

    # ---- cost of building one specialization (compile/warmup) ----
    t = ts[("float16", 4096 * 4096)]
    t0 = time.perf_counter()
    _ = k_nomask.warmup(t, BLOCK=8192, grid=(1,))[(2048, 1, 1)]
    OUT["notes"]["warmup_8192_fp16_s"] = time.perf_counter() - t0
    save()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        OUT["errors"].append(traceback.format_exc()[-3000:])
    save()
    print("ERRORS", json.dumps(OUT["errors"], indent=1)[:3000])
