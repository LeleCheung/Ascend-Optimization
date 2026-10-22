"""Target-side diagnosis of Triton per-launch host overhead for a 1-D elementwise op.

Introspects the installed Triton/FlagTree launch path and microbenchmarks
alternative launch forms against the PyTorch reference floor.
Diagnostic only: no ledger round, no performance claim.
"""
import inspect
import json
import os
import time
import traceback
from pathlib import Path

import torch
import torch_npu  # noqa: F401  (registers the npu device)
import triton
import triton.language as tl

ART = Path(os.environ.get("KGS_DEBUG_ARTIFACTS", "."))
ART.mkdir(parents=True, exist_ok=True)

out = {
    "triton_version": triton.__version__,
    "torch_version": torch.__version__,
    "torch_npu_version": getattr(torch_npu, "__version__", "?"),
    "bench": {},
    "errors": {},
    "facts": {},
}


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


def bench_sync(fn, iters=100, warm=10):
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
grid = ((n + 8191) // 8192,)

# ---- introspection ------------------------------------------------------
try:
    from triton.runtime.jit import JITFunction, CompiledKernel

    out["facts"]["jit_run_src"] = inspect.getsource(JITFunction.run)
    out["facts"]["compiled_getitem_src"] = inspect.getsource(CompiledKernel.__getitem__)
    out["facts"]["compiled_attrs"] = [a for a in dir(CompiledKernel) if not a.startswith("__")]
    out["facts"]["compiled_run_src"] = str(getattr(CompiledKernel, "run", None))
except Exception:
    out["errors"]["introspect"] = traceback.format_exc()

try:
    drv = triton.runtime.driver.active
    props = drv.utils.get_device_properties(0)
    out["facts"]["device_properties"] = {
        k: v for k, v in (props.items() if isinstance(props, dict) else
                          [(a, getattr(props, a)) for a in dir(props) if not a.startswith("_")])
    } if not isinstance(props, str) else props
    out["facts"]["driver"] = drv.__class__.__name__
except Exception:
    out["errors"]["device_props"] = traceback.format_exc()

try:
    drv = triton.runtime.driver.active
    s0 = drv.get_current_stream(0)
    out["facts"]["stream"] = str(s0)
    out["bench"]["get_current_stream_us"] = bench(lambda: drv.get_current_stream(0))
except Exception:
    out["errors"]["stream"] = traceback.format_exc()

# ---- launch-form microbenchmarks ---------------------------------------
out["bench"]["noop_us"] = bench(lambda: None)
out["bench"]["numel_us"] = bench(lambda: A.numel())

try:
    out["bench"]["ref_torch_rad2deg_us"] = bench(lambda: torch.rad2deg_(A))
except Exception:
    out["errors"]["ref"] = traceback.format_exc()

try:
    kptr = A.data_ptr()
    out["bench"]["jit_full_us"] = bench(lambda: _k[grid](A, n, BLOCK=8192))
    out["bench"]["jit_full_us_sync"] = bench_sync(lambda: _k[grid](A, n, BLOCK=8192))
except Exception:
    out["errors"]["jit_full"] = traceback.format_exc()

try:
    out["bench"]["jit_full_ptr_us"] = bench(lambda: _k[grid](kptr, n, BLOCK=8192))
except Exception:
    out["errors"]["jit_full_ptr"] = traceback.format_exc()

compiled = None
try:
    compiled = _k.warmup(A, n, BLOCK=8192, grid=grid)
    out["facts"]["compiled_type"] = type(compiled).__name__
    out["facts"]["compiled_metadata"] = str(getattr(compiled, "metadata", None))[:2000]
    runner = compiled[grid]
    out["bench"]["compiled_runner_us"] = bench(lambda: runner(A, n))
    out["bench"]["compiled_runner_us_sync"] = bench_sync(lambda: runner(A, n))
    out["bench"]["compiled_runner_ptr_us"] = bench(lambda: runner(kptr, n))
except Exception:
    out["errors"]["compiled_runner"] = traceback.format_exc()

# raw launcher with a pre-resolved stream (skips get_current_stream + metadata)
try:
    drv = triton.runtime.driver.active
    stream = drv.get_current_stream(0)
    raw = compiled.run
    fn_handle = compiled.function
    meta = getattr(compiled, "packed_metadata", None)
    forms = {}
    for name, call in {
        "no_meta": lambda: raw(1, 1, 1, stream, fn_handle, kptr, n),
        "with_meta": lambda: raw(1, 1, 1, stream, fn_handle, meta, kptr, n),
        "with_hooks": lambda: raw(1, 1, 1, stream, fn_handle, meta, None, None, None, kptr, n),
    }.items():
        try:
            call()
            torch.npu.synchronize()
            forms[name] = bench(call)
        except Exception:
            forms[name] = traceback.format_exc()[-400:]
    out["bench"]["raw_launcher_us"] = forms
except Exception:
    out["errors"]["raw_launcher"] = traceback.format_exc()

# large-tensor view: is the per-call cost still host-bound at 16.7M elements?
try:
    B = torch.randn(4096, 4096, dtype=torch.float16, device="npu:0")
    nb = B.numel()
    gb = ((nb + 8191) // 8192,)
    out["bench"]["jit_full_16M_us"] = bench(lambda: _k[gb](B, nb, BLOCK=8192), iters=50, warm=10)
    out["bench"]["ref_16M_us"] = bench(lambda: torch.rad2deg_(B), iters=50, warm=10)
except Exception:
    out["errors"]["large"] = traceback.format_exc()

# BLOCK sweep on the big tensor (device-side sensitivity)
try:
    B = torch.randn(4096, 4096, dtype=torch.float16, device="npu:0")
    nb = B.numel()
    sweep = {}
    for blk in (2048, 4096, 8192, 16384, 32768, 65536):
        g = ((nb + blk - 1) // blk,)
        try:
            sweep[blk] = bench(lambda g=g, blk=blk: _k[g](B, nb, BLOCK=blk), iters=50, warm=10)
        except Exception as e:
            sweep[blk] = "ERR:" + str(e)[-200:]
    out["bench"]["block_sweep_16M_fp16_us"] = sweep
except Exception:
    out["errors"]["sweep"] = traceback.format_exc()

(ART / "bench.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
print(json.dumps({k: v for k, v in out["bench"].items() if not isinstance(v, dict)}, indent=2, default=str))
print("BLOCK_SWEEP", json.dumps(out["bench"].get("block_sweep_16M_fp16_us"), default=str))
print("ERRORS", list(out["errors"].keys()))
