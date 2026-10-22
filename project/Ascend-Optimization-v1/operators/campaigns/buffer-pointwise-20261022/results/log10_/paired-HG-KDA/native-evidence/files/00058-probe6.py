"""Probe 6: overlap DMA and math via manual double buffering in a chunked loop."""
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
C = 0.4342944819032518


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
        return {"error": repr(exc)[:250]}


spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)
ADDR = cand._hgkda_address_type
ST = cand._resolve_stream(0)


@triton.jit
def k_db(X, N, CHUNK, BLOCK: tl.constexpr):
    """Chunked loop with one tile of prefetch (software double buffering)."""
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * CHUNK
    r = tl.arange(0, BLOCK)
    x0 = tl.load(X + base + r, mask=(base + r) < N, other=1.0)
    for i in tl.range(BLOCK, CHUNK, BLOCK):
        offs = base + i + r
        x1 = tl.load(X + offs, mask=offs < N, other=1.0)
        prev = offs - BLOCK
        tl.store(X + prev, (tl.log(x0) * 0.4342944819032518).to(X.dtype.element_ty),
                 mask=prev < N)
        x0 = x1
    last = base + CHUNK - BLOCK + r
    tl.store(X + last, (tl.log(x0) * 0.4342944819032518).to(X.dtype.element_ty),
             mask=last < N)


@triton.jit
def k_db3(X, N, CHUNK, BLOCK: tl.constexpr):
    """Two tiles of prefetch."""
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * CHUNK
    r = tl.arange(0, BLOCK)
    x0 = tl.load(X + base + r, mask=(base + r) < N, other=1.0)
    x1 = tl.load(X + base + BLOCK + r, mask=(base + BLOCK + r) < N, other=1.0)
    for i in tl.range(2 * BLOCK, CHUNK, BLOCK):
        offs = base + i + r
        x2 = tl.load(X + offs, mask=offs < N, other=1.0)
        p = offs - 2 * BLOCK
        tl.store(X + p, (tl.log(x0) * 0.4342944819032518).to(X.dtype.element_ty),
                 mask=p < N)
        x0 = x1
        x1 = x2
    p1 = base + CHUNK - 2 * BLOCK + r
    tl.store(X + p1, (tl.log(x0) * 0.4342944819032518).to(X.dtype.element_ty), mask=p1 < N)
    p2 = p1 + BLOCK
    tl.store(X + p2, (tl.log(x1) * 0.4342944819032518).to(X.dtype.element_ty), mask=p2 < N)


def mk(kern, t, chunk, block, grid):
    a = ADDR(t.numel())(t.data_ptr())
    compiled = kern.warmup(t, t.numel(), chunk, BLOCK=block, grid=(1,))
    rn = compiled.run
    call, fn, md = rn.launch, compiled.function, compiled.packed_metadata

    def go():
        call(grid, 1, 1, ST(), fn, md, None, None, None, a, t.numel(), chunk, None)
    return go


def make(dtype, n, seed):
    return (torch.randn(n, generator=torch.Generator().manual_seed(seed)).abs() + 0.05).to(dtype).npu()


N = 4096 * 4096
for dname, dtype, seed in (("fp16", torch.float16, 31), ("fp32", torch.float32, 32)):
    r = {}
    t0 = make(dtype, N, seed)
    cur = cand._make_launch(t0, N)
    r["cur_seed"] = bench(lambda: cur(t0, t0.data_ptr()), 200)
    for p in (40, 80, 160, 512):
        for b in (8192, 16384):
            chunk = ((N + p - 1) // p + b - 1) // b * b
            for label, kern in (("db1", k_db), ("db2", k_db3)):
                t = make(dtype, N, seed)
                key = f"{label}_p{p}_b{b}"
                try:
                    go = mk(kern, t, chunk, b, p)
                    r[key] = bench(go, 200)
                    tt = make(dtype, N, seed + 500)
                    ref = torch.log10(tt.float().clone())
                    mk(kern, tt, chunk, b, p)()
                    torch.npu.synchronize()
                    r[key]["max_abs_err"] = float((tt.float() - ref).abs().max().item())
                except Exception as exc:  # noqa
                    r[key] = {"error": repr(exc)[:200]}
    tt = make(dtype, N, seed + 9)
    r["torch"] = bench((lambda t: (lambda: torch.log10_(t)))(tt), 200)
    out[dname] = r

# host-side: does a plain int pointer work and is it cheaper?
try:
    t = make(torch.float16, 1024, 41)
    compiled = cand._log10_nomask.warmup(t, BLOCK=1024, grid=(1,))
    call, fn, md = compiled.run.launch, compiled.function, compiled.packed_metadata
    ptr = t.data_ptr()
    h = {}
    h["intsub"] = bench(lambda: call(1024, 1, 1, ST(), fn, md, None, None, None,
                                     ADDR(1024)(ptr), None), 2000)
    h["plain_int"] = bench(lambda: call(1024, 1, 1, ST(), fn, md, None, None, None, ptr, None), 2000)
    src = (torch.randn(1024, generator=torch.Generator().manual_seed(42)).abs() + 0.05).to(torch.float16).npu()
    ref = torch.log10(src.clone())
    pre = src.data_ptr()
    call(1024, 1, 1, ST(), fn, md, None, None, None, pre, None)
    torch.npu.synchronize()
    h["plain_int_ok"] = float((src.float() - ref.float()).abs().max().item())
    out["host_forms"] = h
except Exception as exc:  # noqa
    out["host_forms"] = {"error": repr(exc)[:300]}

path = os.path.join(ART, "probe6.json")
with open(path, "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1)[:9000])
print("ARTIFACT", path)
