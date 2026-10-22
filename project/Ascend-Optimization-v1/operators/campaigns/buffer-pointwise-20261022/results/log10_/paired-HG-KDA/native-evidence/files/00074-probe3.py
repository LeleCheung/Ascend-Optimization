"""Probe 3: device-time comparison of tile/grid variants via direct compiled launchers."""
import importlib.util
import json
import os
import time

import torch
import torch_npu
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
out = {}


def bench(fn, iters):
    try:
        for _ in range(10):
            fn()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        t1 = time.perf_counter()
        torch.npu.synchronize()
        t2 = time.perf_counter()
        return {"host_us": round((t1 - t0) / iters * 1e6, 3),
                "total_us": round((t2 - t0) / iters * 1e6, 3)}
    except Exception as exc:  # noqa
        return {"error": repr(exc)[:300]}


spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)
ADDR = cand._hgkda_address_type


def mkfast(kernel, t, scalars, consts, grid):
    """Bind one compiled launcher for kernel(X=t, *scalars) with constexpr consts."""
    n = t.numel()
    a = ADDR(n)(t.data_ptr())
    compiled = kernel.warmup(t, *scalars, **consts, grid=(1,))
    rn = compiled.run
    call, fn, md = rn.launch, compiled.function, compiled.packed_metadata
    st = cand._resolve_stream(t.device.index)
    c = consts

    def go():
        call(grid, 1, 1, st(), fn, md, None, None, None, a, *scalars, None)
    return go, c


@triton.jit
def _chunk(X, N, CHUNK, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * CHUNK
    for i in tl.range(0, CHUNK, BLOCK):
        offs = base + i + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        tl.store(X + offs, (tl.log(x) * 0.4342944819032518).to(X.dtype.element_ty), mask=mask)


@triton.jit
def _chunk_ns2(X, N, CHUNK, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * CHUNK
    for i in tl.range(0, CHUNK, BLOCK, num_stages=2):
        offs = base + i + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        tl.store(X + offs, (tl.log(x) * 0.4342944819032518).to(X.dtype.element_ty), mask=mask)


@triton.jit
def _stride_loop(X, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    span = nprog * BLOCK
    for i in tl.range(pid * BLOCK, N, span):
        offs = i + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        tl.store(X + offs, (tl.log(x) * 0.4342944819032518).to(X.dtype.element_ty), mask=mask)


A16L = (torch.randn(4096, 4096, generator=torch.Generator().manual_seed(2)).abs() + 0.05).to(torch.float16).npu()
A32L = (torch.randn(4096, 4096, generator=torch.Generator().manual_seed(3)).abs() + 0.05).to(torch.float32).npu()


def variants(t):
    n = t.numel()
    r = {}
    cur = cand._make_launch(t, n)
    r["cur_seed"] = bench(lambda: cur(t, t.data_ptr()), 300)
    for b in (8192, 16384, 32768):
        try:
            go, _ = mkfast(cand._log10_nomask, t, (), {"BLOCK": b}, (n // b,))
            r[f"tile_b{b}"] = bench(go, 300)
        except Exception as exc:  # noqa
            r[f"tile_b{b}"] = {"error": repr(exc)[:200]}
    for p in (40, 80):
        for b in (8192, 16384):
            chunk = ((n + p - 1) // p + b - 1) // b * b
            try:
                go, _ = mkfast(_chunk, t, (n, chunk), {"BLOCK": b}, (p,))
                r[f"chunk_p{p}_b{b}"] = bench(go, 300)
            except Exception as exc:  # noqa
                r[f"chunk_p{p}_b{b}"] = {"error": repr(exc)[:200]}
    for p in (40, 80):
        for b in (8192, 16384):
            try:
                go, _ = mkfast(_stride_loop, t, (n,), {"BLOCK": b}, (p,))
                r[f"stride_p{p}_b{b}"] = bench(go, 300)
            except Exception as exc:  # noqa
                r[f"stride_p{p}_b{b}"] = {"error": repr(exc)[:200]}
    chunk = ((n + 39) // 40 + 8191) // 8192 * 8192
    try:
        go, _ = mkfast(_chunk_ns2, t, (n, chunk), {"BLOCK": 8192}, (40,))
        r["chunk_ns2_p40_b8192"] = bench(go, 300)
    except Exception as exc:  # noqa
        r["chunk_ns2_p40_b8192"] = {"error": repr(exc)[:200]}
    r["torch"] = bench(lambda: torch.log10_(t), 300)
    return r


out["fp16_4096x4096"] = variants(A16L)
out["fp32_4096x4096"] = variants(A32L)

# numeric check for the chunk form on a small odd length and on 4096x4096
try:
    src = (torch.randn(1025, generator=torch.Generator().manual_seed(7)).abs() + 0.05).to(torch.float16).npu()
    ref = torch.log10(src.clone()).float()
    got = src.clone()
    n = got.numel()
    go, _ = mkfast(_chunk, got, (n, 8192), {"BLOCK": 8192}, (40,))
    go()
    torch.npu.synchronize()
    out["chunk_check_1025"] = {"max_abs_err": float((got.float() - ref).abs().max().item())}
    g2 = (torch.randn(4096, 4096, generator=torch.Generator().manual_seed(8)).abs() + 0.05).to(torch.float16).npu()
    ref2 = torch.log10(g2.clone()).float()
    go2, _ = mkfast(_chunk, g2, (g2.numel(), 419840), {"BLOCK": 8192}, (40,))
    go2()
    torch.npu.synchronize()
    out["chunk_check_4096"] = {"max_abs_err": float((g2.float() - ref2).abs().max().item())}
except Exception as exc:  # noqa
    out["chunk_check"] = repr(exc)[:300]

path = os.path.join(ART, "probe3.json")
with open(path, "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1)[:9000])
print("ARTIFACT", path)
