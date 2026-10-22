"""Probe 7: unrolled multi-tile programs to overlap DMA with log math."""
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
        return {"error": repr(exc)[:250]}


spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)
ADDR = cand._hgkda_address_type
ST = cand._resolve_stream(0)


@triton.jit
def k_u2(X, BLOCK: tl.constexpr):
    base = tl.program_id(0) * (2 * BLOCK)
    r = tl.arange(0, BLOCK)
    x0 = tl.load(X + base + r)
    x1 = tl.load(X + base + BLOCK + r)
    y0 = tl.log(x0) * 0.4342944819032518
    y1 = tl.log(x1) * 0.4342944819032518
    tl.store(X + base + r, y0.to(X.dtype.element_ty))
    tl.store(X + base + BLOCK + r, y1.to(X.dtype.element_ty))


@triton.jit
def k_u4(X, BLOCK: tl.constexpr):
    base = tl.program_id(0) * (4 * BLOCK)
    r = tl.arange(0, BLOCK)
    x0 = tl.load(X + base + r)
    x1 = tl.load(X + base + BLOCK + r)
    x2 = tl.load(X + base + 2 * BLOCK + r)
    x3 = tl.load(X + base + 3 * BLOCK + r)
    y0 = tl.log(x0) * 0.4342944819032518
    y1 = tl.log(x1) * 0.4342944819032518
    y2 = tl.log(x2) * 0.4342944819032518
    y3 = tl.log(x3) * 0.4342944819032518
    tl.store(X + base + r, y0.to(X.dtype.element_ty))
    tl.store(X + base + BLOCK + r, y1.to(X.dtype.element_ty))
    tl.store(X + base + 2 * BLOCK + r, y2.to(X.dtype.element_ty))
    tl.store(X + base + 3 * BLOCK + r, y3.to(X.dtype.element_ty))


@triton.jit
def k_u2c(X, BLOCK: tl.constexpr):
    """2 tiles with compute/store interleaved (compare scheduling order)."""
    base = tl.program_id(0) * (2 * BLOCK)
    r = tl.arange(0, BLOCK)
    x0 = tl.load(X + base + r)
    x1 = tl.load(X + base + BLOCK + r)
    tl.store(X + base + r, (tl.log(x0) * 0.4342944819032518).to(X.dtype.element_ty))
    tl.store(X + base + BLOCK + r, (tl.log(x1) * 0.4342944819032518).to(X.dtype.element_ty))


@triton.jit
def k_db(X, N, CHUNK, BLOCK: tl.constexpr):
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


def mk(kern, t, scalars, block, grid):
    a = ADDR(t.numel())(t.data_ptr())
    compiled = kern.warmup(t, *scalars, BLOCK=block, grid=(1,))
    rn = compiled.run
    call, fn, md = rn.launch, compiled.function, compiled.packed_metadata

    def go():
        call(grid, 1, 1, ST(), fn, md, None, None, None, a, *scalars, None)
    return go


def make(dtype, n, seed):
    return (torch.randn(n, generator=torch.Generator().manual_seed(seed)).abs() + 0.05).to(dtype).npu()


N = 4096 * 4096
for dname, dtype, seed in (("fp16", torch.float16, 51), ("fp32", torch.float32, 52),
                           ("bf16", torch.bfloat16, 53)):
    r = {}
    t0 = make(dtype, N, seed)
    cur = cand._make_launch(t0, N)
    r["cur_seed"] = bench(lambda: cur(t0, t0.data_ptr()), 200)
    for label, kern, nt in (("u2", k_u2, 2), ("u2c", k_u2c, 2), ("u4", k_u4, 4)):
        for b in (4096, 8192, 16384):
            t = make(dtype, N, seed)
            key = f"{label}_b{b}_grid{N // (nt * b)}"
            try:
                go = mk(kern, t, (), b, N // (nt * b))
                r[key] = bench(go, 200)
                tt = make(dtype, N, seed + 500)
                ref = torch.log10(tt.float().clone())
                mk(kern, tt, (), b, N // (nt * b))()
                torch.npu.synchronize()
                r[key]["max_abs_err"] = float((tt.float() - ref).abs().max().item())
            except Exception as exc:  # noqa
                r[key] = {"error": repr(exc)[:300]}
    tt = make(dtype, N, seed + 9)
    r["torch"] = bench((lambda t: (lambda: torch.log10_(t)))(tt), 200)
    out[dname] = r

# full error text for the loop-carried prefetch kernel
try:
    t = make(torch.float16, N, 61)
    mk(k_db, t, (N, 419840), 8192, 40)
    out["k_db"] = "compiled"
except Exception as exc:  # noqa
    out["k_db"] = str(exc)[:2500]

path = os.path.join(ART, "probe7.json")
with open(path, "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1)[:10000])
print("ARTIFACT", path)
