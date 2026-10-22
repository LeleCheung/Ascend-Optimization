"""Round 2 diagnosis: cheapest legal launch path for the Ascend Triton backend.

Dumps the installed launch-path sources and benchmarks pre-compiled launch
forms (CompiledKernel runner, raw launcher, stream resolution) against the
PyTorch reference. Diagnostic only.
"""
import inspect
import json
import os
import time
import traceback
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = Path(os.environ.get("KGS_DEBUG_ARTIFACTS", "."))
ART.mkdir(parents=True, exist_ok=True)

out = {"bench": {}, "errors": {}, "src": {}, "facts": {}}


@triton.jit
def _k(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(x_ptr + offs, (x.to(tl.float32) * 57.29577951308232).to(x.dtype), mask=m)


def bench(fn, iters=300, warm=30):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


n = 1024
A = torch.randn(n, dtype=torch.float16, device="npu:0")
g = (n + 8191) // 8192

# ---- sources -----------------------------------------------------------
try:
    import triton.spec.ascend.runtime.jit as ajit
    import triton.backends.ascend.driver as adrv

    out["src"]["jit_run"] = inspect.getsource(ajit.JITFunction.run)
    out["src"]["driver"] = inspect.getsource(adrv)
except Exception:
    out["errors"]["src"] = traceback.format_exc()

compiled = None
try:
    compiled = _k.warmup(A, n, BLOCK=8192, grid=(g, 1, 1))
    out["facts"]["compiled_type"] = str(type(compiled))
    out["src"]["compiled"] = inspect.getsource(type(compiled))
    out["facts"]["run_type"] = str(type(compiled.run))
    try:
        out["src"]["launcher"] = inspect.getsource(type(compiled.run))
    except Exception:
        out["errors"]["launcher_src"] = traceback.format_exc()[-500:]
    out["facts"]["run_attrs"] = [a for a in dir(compiled.run) if not a.startswith("__")]
except Exception:
    out["errors"]["warmup"] = traceback.format_exc()

# ---- stream resolution costs ------------------------------------------
try:
    out["bench"]["drv_get_current_stream_us"] = bench(lambda: triton.runtime.driver.active.get_current_stream(0))
except Exception:
    out["errors"]["drv_stream"] = traceback.format_exc()
try:
    out["bench"]["torch_current_stream_us"] = bench(lambda: torch.npu.current_stream())
except Exception:
    out["errors"]["torch_stream"] = traceback.format_exc()
try:
    out["bench"]["torch_stream_param_us"] = bench(lambda: torch.npu.current_stream()._as_parameter_)
except Exception:
    out["errors"]["torch_stream_param"] = traceback.format_exc()
try:
    out["bench"]["torch_stream_id_us"] = bench(lambda: torch.npu.current_stream().stream_id)
except Exception:
    out["errors"]["torch_stream_id"] = traceback.format_exc()

out["bench"]["noop_us"] = bench(lambda: None)
out["bench"]["ref_tight_us"] = bench(lambda: torch.rad2deg_(A))
out["bench"]["jit_full_us"] = bench(lambda: _k[(g,)](A, n, BLOCK=8192))

# ---- pre-compiled forms ------------------------------------------------
runner3 = None
try:
    runner3 = compiled[(g, 1, 1)]
    out["bench"]["runner3_us"] = bench(lambda: runner3(A, n))
except Exception:
    out["errors"]["runner3"] = traceback.format_exc()

try:
    st = triton.runtime.driver.active.get_current_stream(0)
    out["bench"]["runner3_stream_us"] = bench(lambda: runner3(A, n, stream=st))
except Exception:
    out["errors"]["runner3_stream"] = traceback.format_exc()

# raw C launcher: probe the exact 12-argument shape
raw = getattr(compiled, "run", None)
try:
    st = triton.runtime.driver.active.get_current_stream(0)
    fn = compiled.function
    meta = compiled.packed_metadata
    lp = {
        "gscratch": lambda: raw(g, 1, 1, st, fn, meta, None, None, None, None, A, n),
        "trailing": lambda: raw(g, 1, 1, st, fn, meta, None, None, None, A, n, None),
        "gscratch_ptr": lambda: raw(g, 1, 1, st, fn, meta, None, None, None, 0, A, n),
    }
    forms = {}
    for name, call in lp.items():
        try:
            call()
            torch.npu.synchronize()
            forms[name] = bench(call)
        except Exception:
            forms[name] = traceback.format_exc().strip().splitlines()[-1][:200]
    out["bench"]["raw_us"] = forms
except Exception:
    out["errors"]["raw"] = traceback.format_exc()

# ---- realistic fast-path prototypes ------------------------------------
protos = {}


def proto_runner(A, _cache={}):
    nn = A.numel()
    if nn == 0:
        return A
    key = A.dtype
    r = _cache.get(key)
    if r is None:
        r = _cache[key] = compiled[(g, 1, 1)]
    r(A, nn)
    return A


def proto_runner_stream(A, _cache={}):
    nn = A.numel()
    if nn == 0:
        return A
    st = triton.runtime.driver.active.get_current_stream(0)
    r = _cache.get(A.dtype)
    if r is None:
        r = _cache[A.dtype] = compiled[(g, 1, 1)]
    r(A, nn, stream=st)
    return A


for name, fn in (("proto_runner", proto_runner), ("proto_runner_stream", proto_runner_stream)):
    try:
        fn(A)
        torch.npu.synchronize()
        protos[name] = bench(lambda fn=fn: fn(A))
    except Exception:
        protos[name] = traceback.format_exc()[-300:]
out["bench"]["proto_us"] = protos

# ---- large-tensor device time (host-overhead-free) ---------------------
try:
    B = torch.randn(4096, 4096, dtype=torch.float16, device="npu:0")
    nb = B.numel()
    big = {}
    for blk in (8192, 16384, 32768):
        gg = (nb + blk - 1) // blk
        try:
            ck = _k.warmup(B, nb, BLOCK=blk, grid=(gg, 1, 1))
            r = ck[(gg, 1, 1)]
            r(B, nb)
            torch.npu.synchronize()
            ts = []
            for _ in range(20):
                torch.npu.synchronize()
                t0 = time.perf_counter()
                r(B, nb)
                torch.npu.synchronize()
                ts.append((time.perf_counter() - t0) * 1e6)
            ts.sort()
            big[blk] = {"sync_device_us": ts[len(ts) // 2]}
        except Exception:
            big[blk] = traceback.format_exc().strip().splitlines()[-1][:200]
    big["ref_sync_device_us"] = None
    ts = []
    for _ in range(20):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        torch.rad2deg_(B)
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    big["ref_sync_device_us"] = ts[len(ts) // 2]
    out["bench"]["big_device_us"] = big
except Exception:
    out["errors"]["big"] = traceback.format_exc()

(ART / "bench2.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
print(json.dumps({k: v for k, v in out["bench"].items() if not isinstance(v, (dict, str))}, indent=2))
print("PROTO", json.dumps(out["bench"].get("proto_us"), default=str))
print("RAW", json.dumps(out["bench"].get("raw_us"), default=str))
print("BIG", json.dumps(out["bench"].get("big_device_us"), default=str))
print("ERRORS", list(out["errors"].keys()))
