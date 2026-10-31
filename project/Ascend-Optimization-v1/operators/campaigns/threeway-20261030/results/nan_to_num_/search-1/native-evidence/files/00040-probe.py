"""Bounded target-side probe: attribute the ~16.3us small-case floor.

Measures, on the target device, for fp16/bf16/fp32 tensors:
  * seed run() per-call wall time (host bound loop) and device time by npu events
  * torch reference A.nan_to_num_() in the same loops
  * prebuilt compiled-launch call alone (no Python dispatch)
  * minimal-arg prebuilt launch (constants baked as constexpr)
  * nop kernel launch (launch floor)
  * pure Python metadata work / stream reads / empty loop
  * large-case device time for several kernel variants (fp16, bf16, fp32)

Writes a compact JSON artifact. Read-only w.r.t. the workspace.
"""

import json
import os
import sys
import time

sys.path.insert(0, "tmp")

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

import main as cand

DEV = torch.device("npu:0")
ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
LIMITS = cand._DEFAULT_LIMITS


@triton.jit
def _clamp_const(a_ptr, POS: tl.constexpr, NEG: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a_ptr + offs)
    y = tl.maximum(tl.minimum(x, POS), NEG)
    y = tl.where(x != x, 0.0, y)
    tl.store(a_ptr + offs, y)


@triton.jit
def _nop(a_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    if tl.sum(m.to(tl.int32)) < 0:
        tl.store(a_ptr + offs, 0.0, mask=m)


def make(dtype, shape):
    g = torch.Generator().manual_seed(0)
    return torch.randn(shape, generator=g).to(device=DEV, dtype=dtype)


def bench_loop(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    t2 = time.perf_counter()
    return {"host_us": (t1 - t0) / iters * 1e6,
            "wall_incl_sync_us": (t2 - t0) / iters * 1e6}


def bench_events(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    e0 = torch.npu.Event(enable_timing=True)
    e1 = torch.npu.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.npu.synchronize()
    return e1.elapsed_time(e0) / iters * 1e3  # us


def safe(fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return {"error": "%s: %s" % (type(exc).__name__, exc)}


def _build_clamp_const(A, dt, block=1024):
    """Prebuilt launcher for the constants-baked clamp (1 runtime arg)."""
    lim = LIMITS[dt]
    ck = _clamp_const.warmup(A, POS=lim, NEG=-lim, BLOCK=block, grid=(1,))
    launcher, fn, md = ck.run.launch, ck.function, ck.packed_metadata
    stg = lambda: torch_npu._C._npu_getCurrentRawStreamNoWait(0)
    ptr = A.data_ptr()
    out = {"one_arg_wall": bench_loop(
        lambda: launcher((1,), 1, 1, stg(), fn, md, None, None, None,
                         ptr, lim, lim, block))}
    return out


res = {}

# ---------------------------------------------------------------- small cases
small = [(torch.float16, [1024]), (torch.bfloat16, [1024]),
         (torch.float32, [1024]), (torch.float16, [1048576]),
         (torch.bfloat16, [65536]), (torch.float32, [1048576])]
for dt, shape in small:
    A = make(dt, shape)
    key = "%s%s" % (str(dt).split(".")[-1], shape)
    cand.run(A)  # build plan
    torch.npu.synchronize()
    r = {}
    r["seed_run"] = bench_loop(lambda: cand.run(A))
    r["torch_ref"] = bench_loop(lambda: A.nan_to_num_(None, None, None))
    r["seed_run_dev_us"] = bench_events(lambda: cand.run(A), iters=50)
    plan = cand._DEFAULT_PLANS[(dt, A.numel(), A.get_device())]
    if plan is not cand._FAIL:
        launch, fn, md, grid, block, getstream = plan
        ptr, n = A.data_ptr(), A.numel()
        lim = LIMITS[dt]
        r["prebuilt_launch"] = bench_loop(
            lambda: launch(grid, 1, 1, getstream(), fn, md, None, None, None,
                           ptr, n, 0.0, lim, -lim, block))
        r["stream_only"] = bench_loop(lambda: getstream(), iters=2000)
    r["meta_only"] = bench_loop(
        lambda: (A.dtype, A.numel(), A.data_ptr(), A.is_contiguous(),
                 A.get_device()), iters=2000)
    res[key] = r

res["empty_loop"] = bench_loop(lambda: None, iters=2000)
res["current_stream_obj"] = bench_loop(lambda: torch.npu.current_stream().npu_stream,
                                       iters=2000)

# -------------------------------------------------------- launch cost floor
A = make(torch.float16, [1024])
try:
    ck = _nop.warmup(A, A.numel(), BLOCK=1024, grid=(1,))
    launcher, fn, md = ck.run.launch, ck.function, ck.packed_metadata
    st = torch_npu._C._npu_getCurrentRawStreamNoWait(0)
    stg = lambda: torch_npu._C._npu_getCurrentRawStreamNoWait(0)
    ptr, n = A.data_ptr(), A.numel()
    res["nop_prebuilt_launch"] = bench_loop(
        lambda: launcher((1,), 1, 1, stg(), fn, md, None, None, None,
                         ptr, n, 1024))
    res["nop_prebuilt_launch_fixed_stream"] = bench_loop(
        lambda: launcher((1,), 1, 1, st, fn, md, None, None, None,
                         ptr, n, 1024))
except Exception as exc:  # noqa: BLE001
    res["nop_prebuilt_launch"] = {"error": "%s: %s" % (type(exc).__name__, exc)}

# normal JIT dispatch of the nop kernel (for comparison with direct launch)
res["nop_jit_dispatch"] = safe(lambda: bench_loop(
    lambda: _nop[(1,)](A, A.numel(), BLOCK=1024)))

# minimal-arg prebuilt launch (pos/neg/nan baked as constexpr)
res["clamp_const_build"] = safe(lambda: _build_clamp_const(A, torch.float16))

# -------------------------------------------------------- large-case devices
large = {}
for dt in (torch.float16, torch.bfloat16, torch.float32):
    A = make(dt, [4096, 4096])
    n = A.numel()
    lim = LIMITS[dt]
    key = str(dt).split(".")[-1]
    d = {}
    for name, kern, block in (("clamp16384", cand._nan_to_num_clamp, 16384),
                              ("clamp8192", cand._nan_to_num_clamp, 8192),
                              ("clamp32768", cand._nan_to_num_clamp, 32768),
                              ("stage8192", cand._nan_to_num_stage, 8192)):
        grid = (triton.cdiv(n, block),)
        d[name] = safe(lambda kern=kern, block=block, grid=grid: bench_events(
            lambda: kern[grid](A, n, 0.0, lim, -lim, BLOCK=block), iters=30))
    cand.run(A)
    d["seed_run_host_us"] = bench_loop(lambda: cand.run(A), iters=100)
    d["torch_ref_host_us"] = bench_loop(lambda: A.nan_to_num_(None, None, None),
                                        iters=100)
    large[key] = d
res["large_device_us"] = large

path = os.path.join(ART, "probe.json")
with open(path, "w") as fh:
    json.dump(res, fh, indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
print("wrote", path)
