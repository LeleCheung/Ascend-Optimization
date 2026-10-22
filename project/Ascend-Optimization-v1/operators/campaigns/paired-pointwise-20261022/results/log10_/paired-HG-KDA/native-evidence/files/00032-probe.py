"""Target-side diagnosis for log10_: host launch-path breakdown + large-shape tiling sweep.

Diagnostic only; not a candidate. No environment mutation.
"""
import json
import os
import statistics
import sys
import time

import torch
import torch_npu
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
R = {"checks": {}, "host": {}, "bench": {}, "sweep": {}, "seed": {}}


def save():
    path = os.path.join(ART, "probe.json")
    with open(path, "w") as fh:
        json.dump(R, fh, indent=1, default=str)
    print("WROTE", path)


dev = torch.npu.current_device()
R["checks"]["device_index"] = dev
try:
    from triton.runtime import driver
    pr = driver.active.utils.get_device_properties(dev)
    R["checks"]["props"] = ({str(k): str(v) for k, v in pr.items()}
                            if isinstance(pr, dict) else str(pr))
except Exception as exc:  # noqa: BLE001
    R["checks"]["props"] = "ERR %r" % (exc,)
R["checks"]["has_nowait_stream"] = hasattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait")
R["checks"]["torch"] = torch.__version__
R["checks"]["torch_npu"] = getattr(torch_npu, "__version__", "?")

sys.path.insert(0, "tmp")
sys.path.insert(0, ".")
import main as cand  # noqa: E402


def median_call(fn, iters=200, warm=30):
    """median of (call + device sync) — per-call wall time."""
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return statistics.median(ts) * 1e6


def batch_mean(fn, iters=200, warm=30):
    """amortized host enqueue rate over an unsynced batch."""
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def timeit(fn, iters=20000):
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e6


SHAPES = [
    ("float16", 1024), ("float16", 65536), ("float16", 1 << 20), ("float16", 4096 * 4096),
    ("bfloat16", 1024), ("bfloat16", 65536), ("bfloat16", 1 << 20), ("bfloat16", 4096 * 4096),
    ("float32", 1024), ("float32", 65536), ("float32", 1 << 20), ("float32", 4096 * 4096),
]

g = torch.Generator().manual_seed(1729)
tensors = {}
for dt, n in SHAPES:
    a = (torch.randn(n, generator=g).abs() + 0.05).to(getattr(torch, dt)).npu()
    tensors[(dt, n)] = a

# ---- 1. reference vs seed candidate, both timing methodologies -------------
for dt, n in SHAPES:
    a = tensors[(dt, n)]
    key = "%s-%d" % (dt, n)
    ref = a.clone()
    ent = {}
    ent["ref_median"] = median_call(lambda: torch.log10_(ref), iters=100)
    ent["ref_batch"] = batch_mean(lambda: torch.log10_(ref), iters=100)
    ent["cand_median"] = median_call(lambda: cand.run(a), iters=100)
    ent["cand_batch"] = batch_mean(lambda: cand.run(a), iters=100)
    ent["ref_over_cand_median"] = ent["ref_median"] / ent["cand_median"]
    R["bench"][key] = ent
save()

# ---- 2. host path breakdown on a small tensor -----------------------------
a = tensors[("float32", 1024)]
cand.run(a)
launch = next(iter(cand._raw_cache.values()))
ptr = a.data_ptr()
grid = (1,)

# recover bound pieces for the raw pybind call
block = cand._pick_block(a.numel())
compiled = cand._log10_nomask.warmup(a, BLOCK=block, grid=(1,))
launcher = compiled.run
call = launcher.launch
function = compiled.function
metadata = compiled.packed_metadata
getter = torch_npu._C._npu_getCurrentRawStreamNoWait
device = a.device.index
stream = lambda: getter(device)  # noqa: E731

R["host"]["launcher_type"] = type(launcher).__name__
R["host"]["launcher_attrs"] = sorted(x for x in dir(launcher) if not x.startswith("__"))
R["host"]["md_type"] = type(metadata).__name__

R["host"]["run_full"] = median_call(lambda: cand.run(a), iters=300)
R["host"]["launch_only"] = median_call(lambda: launch(a, ptr), iters=300)
R["host"]["pybind_only"] = median_call(
    lambda: call(grid, 1, 1, stream(), function, metadata, None, None, None, ptr, None),
    iters=300)
R["host"]["pybind_fixed_stream"] = median_call(
    lambda: call(grid, 1, 1, getter(device), function, metadata, None, None, None, ptr, None),
    iters=300)
R["host"]["stream_only_us"] = timeit(lambda: getter(device), iters=20000)
R["host"]["numel_us"] = timeit(a.numel, iters=20000)
R["host"]["data_ptr_us"] = timeit(a.data_ptr, iters=20000)
R["host"]["is_contiguous_us"] = timeit(a.is_contiguous, iters=20000)
R["host"]["device_index_us"] = timeit(lambda: a.device.index, iters=20000)
R["host"]["dtype_us"] = timeit(lambda: a.dtype, iters=20000)
R["host"]["key_tuple_us"] = timeit(
    lambda: (a.dtype, a.numel(), a.data_ptr() % 16, a.device.index), iters=20000)
R["host"]["torch_log10_us"] = median_call(lambda: torch.log10_(a), iters=300)
R["host"]["sync_us"] = timeit(lambda: torch.npu.synchronize(), iters=2000)
save()

# ---- 3. large-shape tiling sweep -----------------------------------------


@triton.jit
def _gs(X, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    for s in range(pid * BLOCK, N, step):
        offs = s + tl.arange(0, BLOCK)
        m = offs < N
        x = tl.load(X + offs, mask=m)
        y = tl.log(x.to(tl.float32)) * 0.4342944819032518
        tl.store(X + offs, y.to(X.dtype.element_ty), mask=m)


@triton.jit
def _chunk(X, N, NCHUNK, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * NCHUNK
    for s in range(start, start + NCHUNK, BLOCK):
        offs = s + tl.arange(0, BLOCK)
        m = offs < N
        x = tl.load(X + offs, mask=m)
        y = tl.log(x.to(tl.float32)) * 0.4342944819032518
        tl.store(X + offs, y.to(X.dtype.element_ty), mask=m)


SWEEP = {
    "float16": (4096 * 4096, [8192, 16384, 32768], [20, 40, 80]),
    "float32": (4096 * 4096, [4096, 8192, 16384], [20, 40, 80]),
}

for dt, (n, blocks, grids) in SWEEP.items():
    a = tensors[(dt, n)]
    R["sweep"][dt] = {"seed_run": median_call(lambda: cand.run(a), iters=60),
                      "ref": median_call(lambda: torch.log10_(a), iters=60),
                      "gs": {}, "chunk": {}}
    for b in blocks:
        for gp in grids:
            tag = "g%d_b%d" % (gp, b)
            try:
                R["sweep"][dt]["gs"][tag] = median_call(
                    lambda b=b, gp=gp: _gs[(gp,)](a, n, BLOCK=b), iters=60)
            except Exception as exc:  # noqa: BLE001
                R["sweep"][dt]["gs"][tag] = "ERR %r" % (exc,)
            nchunk = -(-n // gp)
            try:
                R["sweep"][dt]["chunk"][tag] = median_call(
                    lambda b=b, gp=gp, nc=nchunk: _chunk[(gp,)](a, n, nc, BLOCK=b), iters=60)
            except Exception as exc:  # noqa: BLE001
                R["sweep"][dt]["chunk"][tag] = "ERR %r" % (exc,)
    save()

save()
print("DONE")
