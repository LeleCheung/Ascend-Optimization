"""Bounded target probe: decompose the per-call cost of the log10_ candidate."""
import importlib.util
import json
import os
import sys
import time

import torch
import torch_npu
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
out = {}


def jput(k, v):
    out[k] = v


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
        return {"error": repr(exc)[:400]}


# ---------------------------------------------------------------- environment
try:
    from triton.runtime import driver
    p = driver.active.utils.get_device_properties(0)
    jput("props", {str(k): (int(v) if isinstance(v, (int, float)) else str(v))
                   for k, v in dict(p).items()})
except Exception as exc:  # noqa
    jput("props", {"error": repr(exc)[:400]})

spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)

A16 = (torch.randn(1024, generator=torch.Generator().manual_seed(1)).abs() + 0.05).to(torch.float16).npu()
A16L = (torch.randn(4096, 4096, generator=torch.Generator().manual_seed(2)).abs() + 0.05).to(torch.float16).npu()
A32L = (torch.randn(4096, 4096, generator=torch.Generator().manual_seed(3)).abs() + 0.05).to(torch.float32).npu()

# ------------------------------------------------------- launcher introspection
n = A16.numel()
compiled = cand._log10_nomask.warmup(A16, BLOCK=1024, grid=(1,))
runner = compiled.run
info = {
    "runner_type": str(type(runner)),
    "compile_only": bool(getattr(runner, "compile_only", None)),
    "enable_msprof_register_tensor": bool(getattr(runner, "enable_msprof_register_tensor", None)),
    "debug_enabled": bool(getattr(getattr(runner, "metadata", None), "debug_enabled", None)),
    "has_launch": hasattr(runner, "launch"),
    "has_run": hasattr(runner, "run"),
    "metadata_keys": str(list(getattr(runner, "metadata", {}).keys())) if hasattr(getattr(runner, "metadata", None), "keys") else "n/a",
}
try:
    import inspect
    info["launch_sig"] = str(inspect.signature(runner.launch))
except Exception as exc:  # noqa
    info["launch_sig"] = repr(exc)[:200]
try:
    import inspect
    info["runner_run_src"] = inspect.getsource(type(runner).run)[:1500]
except Exception as exc:  # noqa
    info["runner_run_src"] = repr(exc)[:200]
jput("launcher", info)

# --------------------------------------------------------- fast-path timing bits
ptr = A16.data_ptr()
grid = n // 1024
call = runner.launch
function = compiled.function
metadata = compiled.packed_metadata
try:
    raw_stream = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
    jput("raw_stream_api", raw_stream is not None)
except Exception:  # noqa
    jput("raw_stream_api", "err")
stream_get = cand._resolve_stream(A16.device.index)
stream_val = stream_get()
addr = cand._hgkda_address_type(n)(ptr)


def raw_call():
    call(grid, 1, 1, stream_val, function, metadata, None, None, None, addr, None)


m = {}
m["nop"] = bench(lambda: None, 5000)
m["data_ptr"] = bench(lambda: A16.data_ptr(), 5000)
m["numel"] = bench(lambda: A16.numel(), 5000)
m["is_contig"] = bench(lambda: A16.is_contiguous(), 5000)
m["dtype_cmp"] = bench(lambda: A16.dtype is torch.float16, 5000)
m["current_stream"] = bench(lambda: torch.npu.current_stream(), 3000)
m["raw_stream_fn"] = bench(stream_get, 5000)
m["make_addr"] = bench(lambda: cand._hgkda_address_type(n)(ptr), 5000)
m["raw_call"] = bench(raw_call, 3000)
m["triton_dispatch"] = bench(lambda: cand._log10_nomask[grid](A16, BLOCK=1024), 3000)
launch = cand._make_launch(A16, n)
m["cand_launch_closure"] = bench(lambda: launch(A16, ptr), 3000)
m["cand_run"] = bench(lambda: cand.run(A16), 3000)
torch.npu.synchronize()
m["torch_log10_small"] = bench(lambda: torch.log10_(A16), 3000)
jput("small_1024_fp16", m)

# correctness of the raw fast path on a fresh tensor
try:
    src = (torch.randn(4096, generator=torch.Generator().manual_seed(7)).abs() + 0.05).to(torch.float16).npu()
    ref = torch.log10(src.clone())
    got = src.clone()
    before = got.data_ptr()
    cand.run(got)
    torch.npu.synchronize()
    jput("raw_path_check", {
        "same_storage": got.data_ptr() == before,
        "max_abs_err": float((got.float() - ref.float()).abs().max().item()),
    })
except Exception as exc:  # noqa
    jput("raw_path_check", {"error": repr(exc)[:300]})

# --------------------------------------------------------------- large cases
@triton.jit
def _chunk(X, N, CHUNK, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * CHUNK
    for i in tl.range(0, CHUNK, BLOCK):
        offs = base + i + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        tl.store(X + offs, (tl.log(x) * 0.4342944819032518).to(X.dtype.element_ty), mask=mask)


def chunked(t, nprog, block):
    n = t.numel()
    chunk = ((n + nprog - 1) // nprog + block - 1) // block * block
    return lambda: _chunk[nprog](t, n, chunk, BLOCK=block)


for name, t in (("fp16_4096x4096", A16L), ("fp32_4096x4096", A32L)):
    n = t.numel()
    r = {}
    r["torch_log10"] = bench(lambda: torch.log10_(t), 400)
    r["cand_run"] = bench(lambda: cand.run(t), 400)
    for block in (8192, 16384):
        r[f"triton_block{block}"] = bench(
            lambda b=block: cand._log10_nomask[(n + b - 1) // b](t, BLOCK=b), 400)
    for nprog in (20, 40, 80):
        for block in (8192, 16384):
            r[f"chunk_p{nprog}_b{block}"] = bench(chunked(t, nprog, block), 400)
    jput(name, r)

path = os.path.join(ART, "probe.json")
with open(path, "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1)[:12000])
print("ARTIFACT", path)
