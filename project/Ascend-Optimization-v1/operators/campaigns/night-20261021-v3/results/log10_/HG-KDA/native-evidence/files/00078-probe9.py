"""Pick the best device-side tiling for the large shapes with the fixed (no-wait)
launch path, and re-validate numerics of the final candidate."""
import json
import os
import sys
import time
import traceback

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.getcwd(), "tmp"))
import main as cand  # noqa: E402

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = {"bench": {}, "notes": {}, "errors": []}
DEV = "npu:0"
N = 4096 * 4096


def save():
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "probe9.json"), "w") as f:
        json.dump(OUT, f, indent=1, default=str)


@triton.jit
def k_nomask(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.log(x.to(tl.float32)) * 0.4342944819032518
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_persist(X, BLOCK: tl.constexpr, NITER: tl.constexpr):
    """Fixed iteration count; host guarantees grid*BLOCK*NITER == numel."""
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    for i in tl.static_range(NITER):
        offs = (pid + i * npg) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        y = tl.log(x.to(tl.float32)) * 0.4342944819032518
        tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_persist_block(X, BLOCK: tl.constexpr, NITER: tl.constexpr):
    """One contiguous run of NITER*BLOCK per program (better locality per core)."""
    pid = tl.program_id(0)
    base = pid * (BLOCK * NITER)
    for i in tl.static_range(NITER):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        y = tl.log(x.to(tl.float32)) * 0.4342944819032518
        tl.store(X + offs, y.to(X.dtype.element_ty))


def host_time(fn, iters=200):
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
        h, a = host_time(fn)
        e["host_us"], e["amortized_us"] = h, a
    except Exception as exc:  # noqa: BLE001
        e["launch_err"] = f"{type(exc).__name__}: {str(exc)[:200]}"
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

    dev = driver.active.get_current_device()
    nowait = torch_npu._C._npu_getCurrentRawStreamNoWait
    st = int(nowait(dev))
    OUT["notes"]["stream_matches"] = st == int(torch.npu.current_stream().npu_stream)

    def launcher(jitfn, t, block, grid, niter=None):
        if niter is None:
            ck = jitfn.warmup(t, BLOCK=block, grid=(1,))
            pyb = ck.run.launch
            return lambda tt: pyb(grid, 1, 1, nowait(dev), ck.function,
                                  ck.packed_metadata, None, None, None, tt, None)
        ck = jitfn.warmup(t, BLOCK=block, NITER=niter, grid=(1,))
        pyb = ck.run.launch
        return lambda tt: pyb(grid, 1, 1, nowait(dev), ck.function,
                              ck.packed_metadata, None, None, None, tt, None)

    def err_of(t, f):
        a = t.float().clone()
        ref = torch.log10(a)
        b = t.clone()
        f(b)
        torch.npu.synchronize()
        return float((b.float() - ref).abs().max())

    g = torch.Generator().manual_seed(0)
    for dt_name, dt in (("float16", torch.float16), ("bfloat16", torch.bfloat16),
                        ("float32", torch.float32)):
        t = torch.randn(N, generator=g).abs().to(dt).to(DEV) + 0.05
        for block in (16384, 32768):
            f = launcher(k_nomask, t, block, N // block)
            record(f"flat{block}::{dt_name}::16777216", lambda f=f, t=t: f(t),
                   lambda f=f, t=t: err_of(t, f))
        for block, niter in ((16384, 2), (16384, 4), (8192, 4)):
            grid = N // (block * niter)
            f = launcher(k_persist, t, block, grid, niter)
            record(f"persist_stride{block}x{niter}::{dt_name}::16777216",
                   lambda f=f, t=t: f(t), lambda f=f, t=t: err_of(t, f))
            f = launcher(k_persist_block, t, block, grid, niter)
            record(f"persist_contig{block}x{niter}::{dt_name}::16777216",
                   lambda f=f, t=t: f(t), lambda f=f, t=t: err_of(t, f))
        # reference for comparison
        ref_t = t.clone()
        record(f"torch_ref::{dt_name}::16777216", lambda r=ref_t: torch.log10_(r))
    save()

    # current candidate path, final stream accessor
    cand._RUNTIME.clear()
    cand._CACHE.clear()
    t = torch.randn(N, generator=g).abs().to(torch.float16).to(DEV) + 0.05
    cand.run(t)
    OUT["notes"]["cand_stream"] = str(cand._RUNTIME["stream"])
    record("cand_run::float16::16777216", lambda: cand.run(t),
           lambda: err_of(t, lambda x: cand.run(x)))
    small = torch.randn(1024, generator=g).abs().to(torch.float32).to(DEV) + 0.05
    cand.run(small)
    record("cand_run::float32::1024", lambda: cand.run(small))
    record("torch_ref::float32::1024", lambda: torch.log10_(small))
    save()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        OUT["errors"].append(traceback.format_exc()[-3000:])
    save()
    print("ERRORS", json.dumps(OUT["errors"], indent=1)[:2000])
