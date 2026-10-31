"""Probe #2: where does the 10.1us prebuilt Triton launch go, and does
kernel-parameter count matter? Also: native-op launch calibration, and the
bf16/2-byte clamp tile that actually compiles.

Read-only diagnostics; writes probe2.json.
"""

import inspect
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
def _k1(a_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(a_ptr + offs, tl.load(a_ptr + offs))


@triton.jit
def _k3(a_ptr, n, p1, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(a_ptr + offs, tl.load(a_ptr + offs) + p1 * 0.0)


@triton.jit
def _k6(a_ptr, n, p1, p2, p3, p4, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(a_ptr + offs, tl.load(a_ptr + offs) + (p1 + p2 + p3 + p4) * 0.0)


@triton.jit
def _clamp_const(a_ptr, POS: tl.constexpr, NEG: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a_ptr + offs)
    y = tl.maximum(tl.minimum(x, POS), NEG)
    tl.store(a_ptr + offs, tl.where(x != x, 0.0, y))


def bench(fn, iters=300, warmup=30):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e6


def safe(fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return {"error": "%s: %s" % (type(exc).__name__, str(exc)[:600])}


def prebuilt(jitfn, A, grid_n, **kwargs):
    ck = jitfn.warmup(A, *kwargs.pop("args", ()), grid=(1,), **kwargs)
    return (ck.run.launch, ck.function, ck.packed_metadata, int(grid_n))


res = {"triton": {"version": triton.__version__, "file": triton.__file__}}

# ------------------------------------------------------ launcher introspection
A = torch.randn(1024, generator=torch.Generator().manual_seed(0)).to(
    device=DEV, dtype=torch.float16)
cand.run(A)
torch.npu.synchronize()
plan = cand._DEFAULT_PLANS[(torch.float16, 1024, 0)]
meta = {}
try:
    ck = cand._nan_to_num_clamp.warmup(A, 1024, 0.0, 0.0, 0.0, BLOCK=1024, grid=(1,))
    runner = ck.run
    meta["run_type"] = type(runner).__module__ + "." + type(runner).__name__
    meta["run_dir"] = [n for n in dir(runner) if not n.startswith("__")][:60]
    meta["ck_dir"] = [n for n in dir(ck) if not n.startswith("__")][:60]
    try:
        meta["runner_src_file"] = inspect.getsourcefile(type(runner))
    except Exception as exc:  # noqa: BLE001
        meta["runner_src_file"] = str(exc)
    try:
        meta["launch_doc"] = (runner.launch.__doc__ or "")[:800]
    except Exception as exc:  # noqa: BLE001
        meta["launch_doc"] = str(exc)
    md = runner.metadata
    meta["metadata"] = {k: str(getattr(md, k))[:200] for k in dir(md)
                        if not k.startswith("_")}
except Exception as exc:  # noqa: BLE001
    meta["error"] = "%s: %s" % (type(exc).__name__, exc)
res["runner"] = meta

# ascend backend driver source (launch path)
try:
    from triton.runtime import driver as _drv
    mod = type(_drv.active)
    res["active_driver"] = mod.__module__ + "." + mod.__name__
    f = inspect.getsourcefile(mod)
    res["active_driver_file"] = f
    src = inspect.getsource(mod)
    keep = []
    for i, line in enumerate(src.splitlines()):
        if "launch" in line or "aclrt" in line or "def " in line:
            keep.append("%d:%s" % (i, line[:200]))
    res["active_driver_launch_lines"] = keep[:120]
except Exception as exc:  # noqa: BLE001
    res["active_driver"] = "ERR %s: %s" % (type(exc).__name__, exc)
try:
    from triton.backends.ascend.driver import utils as autoc  # noqa: F401
    u = autoc
    res["utils_public"] = [n for n in dir(u) if not n.startswith("_")][:80]
except Exception as exc:  # noqa: BLE001
    res["utils_public"] = "ERR %s" % exc

# --------------------------------------------------- parameter-count sweep
sweep = {}
try:
    e1 = prebuilt(_k1, A, 1, BLOCK=1024)
    stg = lambda: torch_npu._C._npu_getCurrentRawStreamNoWait(0)
    ptr = A.data_ptr()
    sweep["k1_1runtime"] = bench(lambda: e1[0](1, 1, 1, stg(), e1[1], e1[2],
                                               None, None, None, ptr, 1024))
    e3 = prebuilt(_k3, A, 1, args=(1024, 0.0), BLOCK=1024)
    sweep["k3_2runtime"] = bench(lambda: e3[0](1, 1, 1, stg(), e3[1], e3[2],
                                               None, None, None, ptr, 1024, 0.0, 1024))
    e6 = prebuilt(_k6, A, 1, args=(1024, 0.0, 0.0, 0.0, 0.0), BLOCK=1024)
    sweep["k6_5runtime"] = bench(lambda: e6[0](1, 1, 1, stg(), e6[1], e6[2],
                                               None, None, None, ptr, 1024,
                                               0.0, 0.0, 0.0, 0.0, 1024))
    # constants-baked clamp: only the pointer at runtime
    lim = LIMITS[torch.float16]
    ec = prebuilt(_clamp_const, A, 1, POS=lim, NEG=-lim, BLOCK=1024)
    sweep["clamp_const_1arg"] = bench(lambda: ec[0](1, 1, 1, stg(), ec[1], ec[2],
                                                    None, None, None, ptr, lim, -lim, 1024))
    # seed plan launch, for reference in the same process
    launch, fn, md, grid, block, getstream = plan
    sweep["seed_plan_5runtime"] = bench(lambda: launch(grid, 1, 1, getstream(), fn, md,
                                                       None, None, None, ptr, 1024,
                                                       0.0, lim, -lim, block))
    sweep["seed_plan_noargs"] = bench(lambda: launch(grid, 1, 1, getstream(), fn, md,
                                                     None, None, None, ptr, 1024,
                                                     0.0, lim, -lim, block))
except Exception as exc:  # noqa: BLE001
    sweep["error"] = "%s: %s" % (type(exc).__name__, exc)
res["sweep_us"] = sweep

# ------------------------------------------------ native-op launch calibration
res["native_calibration_us"] = {
    "fill_1024": bench(lambda: A.fill_(0.0)),
    "torch_ref_1024": bench(lambda: A.nan_to_num_(None, None, None)),
    "seed_run_1024": bench(lambda: cand.run(A)),
    "add_scalar_out_1024": safe(lambda: bench(lambda: torch.add(A, 1.0, out=A))),
}

# ------------------------------------- which 2-byte clamp tile compiles (bf16)
# compile-only checks: we only need to know if the tile fits UB
bt = {}
for dtype in (torch.bfloat16, torch.float16, torch.float32):
    B = torch.randn(1024).to(device=DEV, dtype=dtype)
    lim = LIMITS[dtype]
    for block in (4096, 8192, 16384):
        key = "%s_B%d" % (str(dtype).split(".")[-1], block)
        bt[key] = safe(lambda block=block, lim=lim, B=B: bool(
            cand._nan_to_num_clamp.warmup(B, 16, 0.0, lim, -lim, BLOCK=block,
                                          grid=(1,)) and True))
res["clamp_compiles"] = bt
bt2 = {}
for dtype in (torch.bfloat16, torch.float16, torch.float32):
    B = torch.randn(1024).to(device=DEV, dtype=dtype)
    lim = LIMITS[dtype]
    for block in (8192, 16384):
        key = "%s_stage_B%d" % (str(dtype).split(".")[-1], block)
        bt2[key] = safe(lambda block=block, lim=lim, B=B: bool(
            cand._nan_to_num_stage.warmup(B, 16, 0.0, lim, -lim, BLOCK=block,
                                          grid=(1,)) and True))
res["stage_compiles"] = bt2

# ------------------------------- large-case device time (wall incl. one sync)
big = {}
for dtype in (torch.float16, torch.bfloat16, torch.float32):
    B = torch.randn((4096, 4096), generator=torch.Generator().manual_seed(1)).to(
        device=DEV, dtype=dtype)
    n = B.numel()
    lim = LIMITS[dtype]
    d = {}
    cand.run(B)
    torch.npu.synchronize()
    d["seed_run_wall"] = bench(lambda: cand.run(B), iters=60)
    d["torch_ref_wall"] = bench(lambda: B.nan_to_num_(None, None, None), iters=60)
    for name, kern, block in (("clamp8192", cand._nan_to_num_clamp, 8192),
                              ("clamp16384", cand._nan_to_num_clamp, 16384),
                              ("stage8192", cand._nan_to_num_stage, 8192),
                              ("stage4096", cand._nan_to_num_stage, 4096)):
        grid = (triton.cdiv(n, block),)
        d[name] = safe(lambda kern=kern, block=block, grid=grid: bench(
            lambda: kern[grid](B, n, 0.0, lim, -lim, BLOCK=block), iters=60))
    big[str(dtype).split(".")[-1]] = d
res["large_wall_us"] = big

path = os.path.join(ART, "probe2.json")
with open(path, "w") as fh:
    json.dump(res, fh, indent=1, default=str)
print(json.dumps(res["sweep_us"], indent=1, default=str))
print(json.dumps(res["native_calibration_us"], indent=1, default=str))
print(json.dumps(res["large_wall_us"], indent=1, default=str))
print(json.dumps(res["clamp_compiles"], indent=1, default=str))
print(json.dumps(res["runner"], indent=1, default=str)[:2000])
print("wrote", path)
