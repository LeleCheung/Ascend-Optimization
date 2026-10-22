"""Probe 8: chunked loop with 2 explicit tiles per iteration (overlap at 40-80 programs)."""
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
        return {"error": repr(exc)[:200]}


def make(dtype, n, seed):
    return (torch.randn(n, generator=torch.Generator().manual_seed(seed)).abs() + 0.05).to(dtype).npu()


spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)
ADDR = cand._hgkda_address_type
ST = cand._resolve_stream(0)
C10 = 0.4342944819032518


@triton.jit
def k_l2(X, N, CHUNK, BLOCK: tl.constexpr):
    """Chunked loop, two tiles per iteration so the second load can overlap."""
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * CHUNK
    r = tl.arange(0, BLOCK)
    for i in tl.range(0, CHUNK, 2 * BLOCK):
        o0 = base + i + r
        o1 = o0 + BLOCK
        x0 = tl.load(X + o0, mask=o0 < N, other=1.0)
        x1 = tl.load(X + o1, mask=o1 < N, other=1.0)
        tl.store(X + o0, (tl.log(x0) * 0.4342944819032518).to(X.dtype.element_ty),
                 mask=o0 < N)
        tl.store(X + o1, (tl.log(x1) * 0.4342944819032518).to(X.dtype.element_ty),
                 mask=o1 < N)


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


def mk(kern, t, scalars, block, grid):
    a = ADDR(t.numel())(t.data_ptr())
    compiled = kern.warmup(t, *scalars, BLOCK=block, grid=(1,))
    rn = compiled.run
    call, fn, md = rn.launch, compiled.function, compiled.packed_metadata

    def go():
        call(grid, 1, 1, ST(), fn, md, None, None, None, a, *scalars, None)
    return go


N = 4096 * 4096
for dname, dtype, seed in (("fp16", torch.float16, 71), ("bf16", torch.bfloat16, 72),
                           ("fp32", torch.float32, 73)):
    r = {}
    t0 = make(dtype, N, seed)
    tv = make(dtype, N, seed + 900)
    ref = torch.log10(tv.float().clone())
    cases = []
    for p in (40, 80):
        for b in (2048, 4096, 8192):
            chunk = ((N + p - 1) // p + 2 * b - 1) // (2 * b) * (2 * b)
            cases.append((f"l2_p{p}_b{b}", k_l2, (N, chunk), b, p))
    cases.append(("u2_b16384", k_u2, (), 16384, N // 32768))
    cases.append(("u4_b8192", k_u4, (), 8192, N // 32768))
    for key, kern, sc, b, g in cases:
        t = make(dtype, N, seed)
        try:
            go = mk(kern, t, sc, b, g)
            r[key] = bench(go, 200)
            tt = make(dtype, N, seed + 900)
            mk(kern, tt, sc, b, g)()
            torch.npu.synchronize()
            r[key]["max_abs_err"] = float((tt.float() - ref).abs().max().item())
        except Exception as exc:  # noqa
            r[key] = {"error": repr(exc)[:200]}
    tt = make(dtype, N, seed + 9)
    r["torch"] = bench((lambda t: (lambda: torch.log10_(t)))(tt), 200)
    out[dname] = r

path = os.path.join(ART, "probe8.json")
with open(path, "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1)[:8000])
print("ARTIFACT", path)
