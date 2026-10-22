"""Probe 2: large-shape device time for tile/grid variants + launcher arg cost."""
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
        for _ in range(20):
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

A16 = (torch.randn(1024, generator=torch.Generator().manual_seed(1)).abs() + 0.05).to(torch.float16).npu()
A16L = (torch.randn(4096, 4096, generator=torch.Generator().manual_seed(2)).abs() + 0.05).to(torch.float16).npu()
A32L = (torch.randn(4096, 4096, generator=torch.Generator().manual_seed(3)).abs() + 0.05).to(torch.float32).npu()

# ------------------------------------------------- launcher source / arity probe
try:
    import inspect
    from triton.backends.ascend.driver import NPULauncher
    src = inspect.getsource(NPULauncher)
    out["npu_launcher_src"] = src[:6000]
except Exception as exc:  # noqa
    out["npu_launcher_src"] = repr(exc)[:300]

n = A16.numel()
compiled = cand._log10_nomask.warmup(A16, BLOCK=1024, grid=(1,))
runner = compiled.run
out["launch_doc"] = str(getattr(runner.launch, "__doc__", ""))[:1500]
call = runner.launch
fn = compiled.function
md = compiled.packed_metadata
st = cand._resolve_stream(A16.device.index)()
addr = cand._hgkda_address_type(n)(A16.data_ptr())
grid = n // 1024

variants = {
    "args11": lambda: call(grid, 1, 1, st, fn, md, None, None, None, addr, None),
    "args10": lambda: call(grid, 1, 1, st, fn, md, None, None, None, addr),
    "args9": lambda: call(grid, 1, 1, st, fn, md, None, None, addr),
    "args8": lambda: call(grid, 1, 1, st, fn, md, addr, None),
    "args6": lambda: call(grid, 1, 1, st, fn, addr),
    "args5": lambda: call(grid, 1, st, fn, addr),
}
res = {}
for k, v in variants.items():
    res[k] = bench(v, 2000)
    torch.npu.synchronize()
out["call_arity"] = res

# ------------------------------------------------------------- large variants


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
def _stride_loop(X, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    span = nprog * BLOCK
    for i in tl.range(pid * BLOCK, N, span):
        offs = i + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        tl.store(X + offs, (tl.log(x) * 0.4342944819032518).to(X.dtype.element_ty), mask=mask)


def chunked(t, nprog, block):
    n = t.numel()
    chunk = ((n + nprog - 1) // nprog + block - 1) // block * block
    return lambda: _chunk[(nprog,)](t, n, chunk, BLOCK=block)


def strided_loop(t, nprog, block):
    n = t.numel()
    return lambda: _stride_loop[(nprog,)](t, n, BLOCK=block)


for name, t in (("fp16_4096x4096", A16L), ("fp32_4096x4096", A32L)):
    n = t.numel()
    r = {}
    r["torch"] = bench(lambda: torch.log10_(t), 400)
    r["cand_run"] = bench(lambda: cand.run(t), 400)
    for block in (8192, 16384, 32768):
        r[f"tile_b{block}"] = bench(
            lambda b=block: cand._log10_nomask[((n + b - 1) // b,)](t, BLOCK=b), 300)
    for nprog in (20, 40, 80):
        for block in (8192, 16384):
            r[f"chunk_p{nprog}_b{block}"] = bench(chunked(t, nprog, block), 300)
    for nprog in (40, 80):
        for block in (8192, 16384):
            r[f"stride_p{nprog}_b{block}"] = bench(strided_loop(t, nprog, block), 300)
    out[name] = r

# correctness of the best-looking candidate forms
try:
    src = (torch.randn(1025, generator=torch.Generator().manual_seed(7)).abs() + 0.05).to(torch.float16).npu()
    ref = torch.log10(src.clone()).float()
    got = src.clone()
    chunked(got, 40, 8192)()
    torch.npu.synchronize()
    a = got.float()
    out["chunk_check_1025"] = {"max_abs_err": float((a - ref).abs().max().item())}
    got2 = (torch.randn(4096, 4096, generator=torch.Generator().manual_seed(8)).abs() + 0.05).to(torch.float16).npu()
    ref2 = torch.log10(got2.clone()).float()
    chunked(got2, 40, 16384)()
    torch.npu.synchronize()
    out["chunk_check_4096"] = {"max_abs_err": float((got2.float() - ref2).abs().max().item())}
except Exception as exc:  # noqa
    out["chunk_check"] = repr(exc)[:300]

path = os.path.join(ART, "probe2.json")
with open(path, "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "npu_launcher_src"}, indent=1)[:9000])
print("LAUNCHER_SRC", out.get("npu_launcher_src", "")[:5000])
print("ARTIFACT", path)
