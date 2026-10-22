"""Probe 6: find the exact backend launcher call signature for a direct
CompiledKernel re-launch, verify it numerically and time it."""
import importlib.util
import inspect
import json
import os
import pathlib
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = pathlib.Path(os.environ.get("KGS_DEBUG_ARTIFACTS", "."))
ART.mkdir(parents=True, exist_ok=True)
R = {}


def load_candidate():
    path = pathlib.Path("tmp/main.py")
    if not path.is_file():
        path = next(pathlib.Path.cwd().rglob("tmp/main.py"))
    spec = importlib.util.spec_from_file_location("cand", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = load_candidate()

src = {}
for name, obj in [("JITFunction.run", getattr(triton.runtime.jit.JITFunction, "run", None)),
                  ("CompiledKernel.init", getattr(triton.compiler.compiler.CompiledKernel, "__init__", None))]:
    try:
        src[name] = inspect.getsource(obj)
    except Exception as exc:
        src[name] = "ERROR: " + str(exc)[:200]
R["sources"] = src


def bench_us(fn, iters=100):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    e0 = torch.npu.Event(enable_timing=True)
    e1 = torch.npu.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.npu.synchronize()
    return e0.elapsed_time(e1) * 1e3 / iters


def cpu_us(fn, iters=200):
    for _ in range(5):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1e6 / iters


n, block = 65536, 512
grid = triton.cdiv(n, block * 4)
x = torch.empty(n, dtype=torch.float32, device="npu:0")
lo, hi = mod._sample_bounds(torch.float32, 0.0, 1.0)
args = (x, n, 12345, 0, 0.0, 1.0, lo, hi)
kern = mod.uniform_kernel[(grid,)](*args, BLOCK=block)
torch.npu.synchronize()
drv = triton.runtime.driver.active
stream = drv.get_current_stream(torch.npu.current_device())

variants = {
    "extra0_after_hooks": lambda: kern.run(grid, 1, 1, stream, kern.function, kern.packed_metadata,
                                           None, None, None, 0, *args),
    "extra0_before_hooks": lambda: kern.run(grid, 1, 1, stream, kern.function, kern.packed_metadata,
                                            0, None, None, None, *args),
    "extra0_before_meta": lambda: kern.run(grid, 1, 1, stream, kern.function, 0, kern.packed_metadata,
                                           None, None, None, *args),
    "two_extra": lambda: kern.run(grid, 1, 1, stream, kern.function, kern.packed_metadata,
                                  None, None, None, 0, 0, *args),
}
out = {}
for name, fn in variants.items():
    try:
        fn()
        out[name] = {"status": "ran", "us": round(bench_us(fn, 50), 2), "cpu_us": round(cpu_us(fn, 100), 2)}
    except Exception as exc:
        out[name] = "ERROR: " + type(exc).__name__ + ": " + str(exc)[:160]

# numeric verification for whichever variant ran
work = [k for k, v in out.items() if isinstance(v, dict)]
if work:
    name = work[0]
    try:
        a = torch.empty(n, dtype=torch.float32, device="npu:0")
        b = torch.empty(n, dtype=torch.float32, device="npu:0")
        mod.uniform_kernel[(grid,)](a, n, 12345, 0, 0.0, 1.0, lo, hi, BLOCK=block)
        variants[name]()
        torch.npu.synchronize()
        out["equal"] = bool(torch.equal(a, b)) or "different-tensor"
    except Exception as exc:
        out["equal"] = "ERROR: " + str(exc)[:200]
out["normal_us"] = round(bench_us(lambda: mod.uniform_kernel[(grid,)](*args, BLOCK=block), 50), 2)
out["normal_cpu_us"] = round(cpu_us(lambda: mod.uniform_kernel[(grid,)](*args, BLOCK=block), 100), 2)
out["stream_cpu_us"] = round(cpu_us(lambda: drv.get_current_stream(torch.npu.current_device()), 200), 2)
out["current_device_cpu_us"] = round(cpu_us(lambda: torch.npu.current_device(), 200), 2)
out["wrapper_us"] = round(bench_us(lambda: mod.run(x, 0.0, 1.0), 100), 2)
R["launch"] = out

(ART / "probe6-uniform.json").write_text(json.dumps(R, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(R, indent=1, ensure_ascii=False)[:7000])
