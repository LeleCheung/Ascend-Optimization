"""Probe 5: copy floor vs log math cost, one kernel per formulation (single runtime arg)."""
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
C = 0.4342944819032518


def mkfast(kernel, t, block, grid):
    a = ADDR(t.numel())(t.data_ptr())
    compiled = kernel.warmup(t, BLOCK=block, grid=(1,))
    rn = compiled.run
    call, fn, md = rn.launch, compiled.function, compiled.packed_metadata

    def go():
        call(grid, 1, 1, ST(), fn, md, None, None, None, a, None)
    return go


@triton.jit
def k_copy(X, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(X + offs, tl.load(X + offs))


@triton.jit
def k_log(X, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    tl.store(X + offs, (tl.log(x) * 0.4342944819032518).to(X.dtype.element_ty))


@triton.jit
def k_log2(X, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    tl.store(X + offs, (tl.log2(x) * 0.3010299956639812).to(X.dtype.element_ty))


@triton.jit
def k_log_h(X, BLOCK: tl.constexpr):
    """log in declared element precision where possible."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.log(x.to(tl.float32) + 1e-30)
    tl.store(X + offs, (y * 0.4342944819032518).to(X.dtype.element_ty))


@triton.jit
def k_copy_chunk(X, N, CHUNK, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * CHUNK
    for i in tl.range(0, CHUNK, BLOCK):
        offs = base + i + tl.arange(0, BLOCK)
        mask = offs < N
        tl.store(X + offs, tl.load(X + offs, mask=mask), mask=mask)


KERNELS = {"copy": k_copy, "log": k_log, "log2": k_log2, "log_h": k_log_h}


def make(dtype, n, seed, flat=True):
    t = (torch.randn(n, generator=torch.Generator().manual_seed(seed)).abs() + 0.05).to(dtype).npu()
    return t


N = 4096 * 4096
for dname, dtype, seed in (("fp16", torch.float16, 11), ("fp32", torch.float32, 12)):
    r = {}
    for label, kern in KERNELS.items():
        for block in (8192, 16384, 32768):
            t = make(dtype, N, seed)
            try:
                go = mkfast(kern, t, block, N // block)
                key = f"{label}_b{block}"
                r[key] = bench(go, 200)
                tt = make(dtype, N, seed + 100)
                ref = torch.log10(tt.float().clone())
                go2 = mkfast(kern, tt, block, N // block)
                go2()
                torch.npu.synchronize()
                d = (tt.float() - ref).abs()
                r[key]["max_abs_err"] = float(d.max().item())
            except Exception as exc:  # noqa
                r[f"{label}_b{block}"] = {"error": repr(exc)[:250]}
    # chunked copy at torch-like granularity (40 programs)
    for block in (8192, 16384):
        t = make(dtype, N, seed)
        chunk = ((N + 39) // 40 + block - 1) // block * block
        try:
            a = ADDR(N)(t.data_ptr())
            compiled = k_copy_chunk.warmup(t, N, chunk, BLOCK=block, grid=(1,))
            rn = compiled.run
            call, fn, md = rn.launch, compiled.function, compiled.packed_metadata

            def go(a=a, call=call, fn=fn, md=md, chunk=chunk, block=block):
                call(40, 1, 1, ST(), fn, md, None, None, None, a, N, chunk, None)
            r[f"copychunk_p40_b{block}"] = bench(go, 200)
        except Exception as exc:  # noqa
            r[f"copychunk_p40_b{block}"] = {"error": repr(exc)[:250]}
    tt = make(dtype, N, seed + 7)
    r["torch"] = bench((lambda t: (lambda: torch.log10_(t)))(tt), 200)
    out[dname] = r

path = os.path.join(ART, "probe5.json")
with open(path, "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1)[:9000])
print("ARTIFACT", path)
