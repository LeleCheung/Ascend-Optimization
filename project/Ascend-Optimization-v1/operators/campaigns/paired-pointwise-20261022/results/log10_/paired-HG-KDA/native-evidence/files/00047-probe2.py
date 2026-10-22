"""Target-side diagnosis #2: host launch-path breakdown + 4096x4096 tiling sweep."""

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
R = {"host": {}, "sweep": {}, "sweep2": {}}


def save():
    path = os.path.join(ART, "probe2.json")
    with open(path, "w") as fh:
        json.dump(R, fh, indent=1, default=str)
    print("WROTE", path)


sys.path.insert(0, "tmp")
sys.path.insert(0, ".")
import main as cand  # noqa: E402

dev = torch.npu.current_device()


def batch_mean(fn, iters=200, warm=30):
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


g = torch.Generator().manual_seed(1729)
a32 = (torch.randn(1024, generator=g).abs() + 0.05).to(torch.float32).npu()

# ---------------- host path breakdown (n = 1024, fp32) ----------------
cand.run(a32)
launch = next(iter(cand._raw_cache.values()))
ptr = a32.data_ptr()
block = cand._pick_block(a32.numel())
compiled = cand._log10_nomask.warmup(a32, BLOCK=block, grid=(1,))
launcher = compiled.run
call = launcher.launch
function = compiled.function
metadata = compiled.packed_metadata
getter = torch_npu._C._npu_getCurrentRawStreamNoWait
device = a32.device.index
gx = 1

R["host"]["launcher_type"] = type(launcher).__name__
R["host"]["launcher_public"] = sorted(x for x in dir(launcher) if not x.startswith("_"))
R["host"]["md_type"] = type(metadata).__name__
R["host"]["md_repr"] = repr(metadata)[:400]
R["host"]["call_doc"] = (call.__doc__ or "")[:800]
R["host"]["call_text_sig"] = getattr(call, "__text_signature__", "") or ""
R["host"]["launcher_doc"] = (type(launcher).__call__.__doc__ or "")[:800]
R["host"]["raw_stream_int"] = getter(device)
R["host"]["grid_int_type"] = str(type(gx))

R["host"]["run_full"] = batch_mean(lambda: cand.run(a32), iters=400)
R["host"]["launch_only"] = batch_mean(lambda: launch(a32, ptr), iters=400)
R["host"]["pybind_ptr"] = batch_mean(
    lambda: call(gx, 1, 1, getter(device), function, metadata, None, None, None, ptr, None),
    iters=400)
R["host"]["pybind_tensor"] = batch_mean(
    lambda: call(gx, 1, 1, getter(device), function, metadata, None, None, None, a32, None),
    iters=400)
R["host"]["launcher_call"] = batch_mean(
    lambda: launcher(gx, 1, 1, getter(device), function, metadata, None, None, None, a32, None),
    iters=400)
R["host"]["pybind_cached_stream"] = batch_mean(
    lambda: call(gx, 1, 1, getter(device), function, metadata, None, None, None, ptr, None),
    iters=400)
R["host"]["stream_only_us"] = timeit(lambda: getter(device), iters=20000)
R["host"]["numel_us"] = timeit(a32.numel, iters=20000)
R["host"]["data_ptr_us"] = timeit(a32.data_ptr, iters=20000)
R["host"]["is_contig_us"] = timeit(a32.is_contiguous, iters=20000)
R["host"]["dev_idx_us"] = timeit(lambda: a32.device.index, iters=20000)
R["host"]["dtype_us"] = timeit(lambda: a32.dtype, iters=20000)
R["host"]["key4_us"] = timeit(
    lambda: (a32.dtype, a32.numel(), a32.data_ptr() % 16, a32.device.index), iters=20000)
R["host"]["torch_log10_np"] = batch_mean(lambda: torch.log10_(a32), iters=400)
R["host"]["torch_log10_fresh"] = batch_mean(
    lambda: torch.log10_(a32.clone()), iters=400)
R["host"]["empty_like"] = timeit(lambda: torch.empty_like(a32), iters=5000)
save()

# ---------------- 4096x4096 tiling sweep ----------------


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


@triton.jit
def _gs_nm(X, N, BLOCK: tl.constexpr):
    """Contiguous-chunk-free grid stride, main-tile loop with a single tail."""
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    s = pid * BLOCK
    while s + BLOCK <= N:
        offs = s + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        y = tl.log(x.to(tl.float32)) * 0.4342944819032518
        tl.store(X + offs, y.to(X.dtype.element_ty))
        s += step
    if s < N:
        offs = s + tl.arange(0, BLOCK)
        m = offs < N
        x = tl.load(X + offs, mask=m)
        y = tl.log(x.to(tl.float32)) * 0.4342944819032518
        tl.store(X + offs, y.to(X.dtype.element_ty), mask=m)


N = 4096 * 4096
SWEEP = {"float16": ([8192, 16384, 32768], [20, 40, 80]),
         "float32": ([4096, 8192, 16384], [20, 40, 80])}
for dt, (blocks, grids) in SWEEP.items():
    a = (torch.randn(N, generator=g).abs() + 0.05).to(getattr(torch, dt)).npu()
    ent = {"seed_run": batch_mean(lambda: cand.run(a), iters=60),
           "ref": batch_mean(lambda: torch.log10_(a), iters=60),
           "gs": {}, "chunk": {}, "gs_nm": {}}
    for b in blocks:
        for gp in grids:
            tag = "g%d_b%d" % (gp, b)
            nchunk = -(-N // gp)
            ent["gs"][tag] = batch_mean(lambda b=b, gp=gp: _gs[(gp,)](a, N, BLOCK=b), iters=60)
            ent["chunk"][tag] = batch_mean(
                lambda b=b, gp=gp, nc=nchunk: _chunk[(gp,)](a, N, nc, BLOCK=b), iters=60)
            ent["gs_nm"][tag] = batch_mean(
                lambda b=b, gp=gp: _gs_nm[(gp,)](a, N, BLOCK=b), iters=60)
    R["sweep"][dt] = ent
    save()

# mid-size check with the likely best configs
for dt in ("float16", "float32"):
    for n in (65536, 1 << 20):
        a = (torch.randn(n, generator=g).abs() + 0.05).to(getattr(torch, dt)).npu()
        e = {"ref": batch_mean(lambda: torch.log10_(a), iters=100),
             "seed": batch_mean(lambda: cand.run(a), iters=100)}
        for gp in (20, 40):
            for b in (4096, 8192):
                e["gs_g%d_b%d" % (gp, b)] = batch_mean(
                    lambda b=b, gp=gp: _gs[(gp,)](a, n, BLOCK=b), iters=100)
        R["sweep2"]["%s-%d" % (dt, n)] = e
save()
print("DONE")
