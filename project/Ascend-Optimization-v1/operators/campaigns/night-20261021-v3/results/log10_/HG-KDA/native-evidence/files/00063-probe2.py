"""Decompose log10_ per-call cost on Ascend: host launch path vs device time,
and sweep tile / mask / persistent variants. Writes incremental JSON."""
import inspect
import json
import os
import time
import traceback

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

OUT = {"bench": {}, "errors": [], "notes": {}}
ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")


def dump():
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "probe2.json"), "w") as f:
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
    """Requires numel % BLOCK == 0 (host guard)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.log(x.to(tl.float32)) * 0.4342944819032518
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_f16(X, BLOCK: tl.constexpr):
    """Compute the log in the input dtype via log2 (fp16 only, no mask)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.log2(x) * 0.30102999566398120
    tl.store(X + offs, y.to(X.dtype.element_ty))


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


def sync_time(fn, iters=100):
    fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
        torch.npu.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e6


def err_vs_ref(x, fn):
    a = x.clone()
    fn(a)
    torch.npu.synchronize()
    ref = torch.log10(x.float()).double()
    got = a.float().double()
    d = (got - ref).abs()
    return float(d.max()), float((d / ref.abs().clamp_min(1e-30)).max())


def record(name, fn, err_fn=None):
    e = {}
    try:
        e["host_us"], e["amortized_us"] = host_time(fn)
        e["sync_us"] = sync_time(fn)
    except Exception as exc:  # noqa: BLE001
        e["launch_err"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    if err_fn is not None:
        try:
            e["abs_err"], e["rel_err"] = err_fn()
        except Exception as exc:  # noqa: BLE001
            e["err"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    OUT["bench"][name] = e
    dump()
    print(name, json.dumps(e, default=str))


def main():
    # ---------- props ----------
    p = {}
    try:
        tp = torch.npu.get_device_properties(0)
        p["torch_props"] = str(tp)
        p["torch_props_dir"] = [k for k in dir(tp) if not k.startswith("_")]
    except Exception as exc:  # noqa: BLE001
        p["torch_props"] = f"ERR {exc}"
    try:
        from triton.runtime import driver

        d = driver.active.utils.get_device_properties(torch.npu.current_device())
        p["triton_props"] = {k: int(v) for k, v in d.items()} if isinstance(d, dict) else str(d)
    except Exception as exc:  # noqa: BLE001
        p["triton_props"] = f"ERR {exc}"
    OUT["props"] = p
    print(json.dumps(p, indent=1)[:1500])
    dump()

    # ---------- tensors ----------
    dev = "npu:0"
    g = torch.Generator().manual_seed(0)
    t = {}
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        for n in (1024, 65536, 1048576, 4096 * 4096):
            base = torch.randn(n, generator=g).abs().to(dt).to(dev) + 0.05
            t[(str(dt).split(".")[-1], n)] = base

    # ---------- reference ----------
    for (dt, n) in (("float16", 1024), ("float16", 1048576), ("float16", 4096 * 4096),
                    ("float32", 1024), ("float32", 1048576), ("float32", 4096 * 4096)):
        x = t[(dt, n)]
        record(f"torch::{dt}::{n}", lambda x=x: torch.log10_(x))
        OUT["bench"][f"torch::{dt}::{n}"]["alias_ok"] = (torch.log10_(x) is x)

    # ---------- launch path decomposition ----------
    x = t[("float32", 4096 * 4096)]
    jit = k_flat
    try:
        OUT["notes"]["jit_dir"] = [a for a in dir(jit) if not a.startswith("_")][:40]
        ck = jit.warmup(x, x.numel(), BLOCK=16384, grid=(1,))
        OUT["notes"]["ck_type"] = type(ck).__name__
        OUT["notes"]["ck_dir"] = [a for a in dir(ck) if not a.startswith("_")][:60]
        try:
            OUT["notes"]["ck_run_sig"] = str(inspect.signature(ck.run))
        except Exception as exc:  # noqa: BLE001
            OUT["notes"]["ck_run_sig"] = f"ERR {exc}"
        grid = (x.numel() // 16384,)
        try:
            runner = ck[grid]
            record("direct_ck_getitem::float32::16777216",
                   lambda r=runner, x=x: r(x, x.numel()))
        except Exception as exc:  # noqa: BLE001
            OUT["errors"].append(f"ck[grid] failed: {type(exc).__name__}: {str(exc)[:400]}")
        try:
            meta = ck.packed_metadata
            import torch_npu  # noqa: F401
            stream = torch.npu.current_stream().npu_stream

            def direct_run(x=x, grid=grid):
                ck.run(grid[0], 1, 1, stream, ck.function, meta, None, None, None, x, x.numel())

            record("direct_ck_run::float32::16777216", direct_run)
        except Exception as exc:  # noqa: BLE001
            OUT["errors"].append(f"ck.run failed: {type(exc).__name__}: {str(exc)[:400]}")
    except Exception as exc:  # noqa: BLE001
        OUT["errors"].append(f"warmup/direct probe failed: {type(exc).__name__}: {str(exc)[:600]}")
    dump()

    # ---------- masked flat sweep ----------
    for dt in ("float16", "float32"):
        for n in (1024, 65536, 1048576, 4096 * 4096):
            x = t[(dt, n)]
            for blk in (1024, 8192, 16384, 32768):
                if n < blk:
                    continue
                grid = (triton.cdiv(n, blk),)
                record(f"flat{blk}::{dt}::{n}",
                       lambda x=x, gr=grid, b=blk: k_flat[gr](x, x.numel(), BLOCK=b))

    # ---------- unmasked flat (numel % BLOCK == 0) ----------
    for dt in ("float16", "float32"):
        for n in (65536, 1048576, 4096 * 4096):
            x = t[(dt, n)]
            for blk in (8192, 16384, 32768):
                if n % blk:
                    continue
                grid = (n // blk,)
                record(f"nomask{blk}::{dt}::{n}",
                       lambda x=x, gr=grid, b=blk: k_nomask[gr](x, BLOCK=b),
                       lambda x=x, gr=grid, b=blk: err_vs_ref(x, lambda a: k_nomask[gr](a, BLOCK=b)))

    # ---------- fp16-native compute ----------
    for n in (1048576, 4096 * 4096):
        x = t[("float16", n)]
        for blk in (16384, 32768):
            grid = (n // blk,)
            record(f"f16native{blk}::float16::{n}",
                   lambda x=x, gr=grid, b=blk: k_f16[gr](x, BLOCK=b),
                   lambda x=x, gr=grid, b=blk: err_vs_ref(x, lambda a: k_f16[gr](a, BLOCK=b)))

    # ---------- persistent ----------
    nvc = 0
    try:
        from triton.runtime import driver

        d = driver.active.utils.get_device_properties(torch.npu.current_device())
        nvc = int(d.get("num_vectorcore") or 0)
    except Exception:  # noqa: BLE001
        nvc = 0
    OUT["notes"]["num_vectorcore"] = nvc
    for dt in ("float16", "float32"):
        x = t[(dt, 4096 * 4096)]
        n = 4096 * 4096
        for mult in (1, 2, 4):
            for blk in (16384, 32768):
                g_ = max(1, min(nvc * mult if nvc else 40 * mult, triton.cdiv(n, blk)))
                niter = triton.cdiv(triton.cdiv(n, blk), g_)
                if g_ * niter * blk < n:
                    continue
                record(f"persist_m{mult}_{blk}::{dt}::{n}",
                       lambda x=x, gg=(g_,), b=blk, it=niter: k_persist[gg](x, x.numel(), it, BLOCK=b),
                       lambda x=x, gg=(g_,), b=blk, it=niter: err_vs_ref(
                           x, lambda a: k_persist[gg](a, a.numel(), it, BLOCK=b)))

    # ---------- bf16 check on the best-looking config ----------
    for n in (1048576, 4096 * 4096):
        x = t[("bfloat16", n)]
        for blk in (16384, 32768):
            grid = (n // blk,)
            record(f"nomask{blk}::bfloat16::{n}",
                   lambda x=x, gr=grid, b=blk: k_nomask[gr](x, BLOCK=b),
                   lambda x=x, gr=grid, b=blk: err_vs_ref(x, lambda a: k_nomask[gr](a, BLOCK=b)))

    dump()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        OUT["errors"].append(traceback.format_exc()[-3000:])
    dump()
    print("ERRORS:", json.dumps(OUT["errors"], indent=1)[:4000])
