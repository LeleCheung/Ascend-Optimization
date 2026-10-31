"""Bounded diagnostic #2b: host-cost breakdown + device kernel-shape comparison."""
import json
import os
import time
import importlib.util

import torch
import torch_npu
import triton
import triton.language as tl

ARTS = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
res = {}

spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)


def bench(fn, iters=200, warm=5):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    t2 = time.perf_counter()
    return {"submit_us": round((t1 - t0) / iters * 1e6, 3),
            "wall_us": round((t2 - t0) / iters * 1e6, 3)}


@triton.jit
def k_tiny(X, Y):
    tl.store(Y + tl.program_id(0), tl.load(X + tl.program_id(0)))


# ---------------- host breakdown ----------------------------------------
x = torch.randn(1024, device="npu:0", dtype=torch.float16)
out = torch.empty_like(x)
n = x.numel()
compiled = cand._fix_contiguous.warmup(x, out, n, BLOCK=256, grid=(1,))
launcher = compiled.run
fn = launcher.launch
fptr, optr = x.data_ptr(), out.data_ptr()
raw = torch_npu._C._npu_getCurrentRawStreamNoWait
stream = raw(0)
gridx = 4
ctiny = k_tiny.warmup(x, out, grid=(1,))
tfn = ctiny.run.launch
res["host"] = {
    "empty_like": bench(lambda: torch.empty_like(x)),
    "empty_shape_dev": bench(lambda: torch.empty(x.shape, dtype=x.dtype, device=x.device)),
    "empty_like_mf": bench(lambda: torch.empty_like(x, memory_format=torch.contiguous_format)),
    "raw_stream_call": bench(lambda: raw(0)),
    "npu_stream": bench(lambda: torch.npu.current_stream(0).npu_stream),
    "launch_only": bench(lambda: fn(gridx, 1, 1, stream, compiled.function,
                                    compiled.packed_metadata, None, None, None,
                                    fptr, optr, n, None)),
    "jit_dispatch": bench(lambda: cand._fix_contiguous[gridx](x, out, n, BLOCK=256)),
    "cand_run": bench(lambda: cand.run(x)),
    "ref_fix": bench(lambda: torch.fix(x)),
    "device_index": bench(lambda: x.device.index),
    "is_contig": bench(lambda: x.is_contiguous()),
    "data_ptr": bench(lambda: x.data_ptr()),
}

# ---------------- device kernel variants --------------------------------
@triton.jit
def _trunc_seed(x):
    xf = x.to(tl.float32)
    t = tl.where(xf >= 0, tl.floor(xf), tl.ceil(xf))
    s = xf.to(tl.uint32, bitcast=True) & 0x80000000
    m = t.to(tl.uint32, bitcast=True) & 0x7fffffff
    return (m | s).to(tl.float32, bitcast=True)


@triton.jit
def _trunc_plain(x):
    xf = x.to(tl.float32)
    return tl.where(xf >= 0, tl.floor(xf), tl.ceil(xf))


@triton.jit
def k_masked_seed(X, Y, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(X + i, i < N, other=0)
    tl.store(Y + i, _trunc_seed(v).to(Y.dtype.element_ty), i < N)


@triton.jit
def k_nomask_seed(X, Y, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(X + i)
    tl.store(Y + i, _trunc_seed(v).to(Y.dtype.element_ty))


@triton.jit
def k_nomask_plain(X, Y, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(X + i)
    tl.store(Y + i, _trunc_plain(v).to(Y.dtype.element_ty))


@triton.jit
def k_nomask_trunc(X, Y, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(X + i)
    tl.store(Y + i, tl.math.trunc(v.to(tl.float32)).to(Y.dtype.element_ty))


@triton.jit
def k_loop_plain(X, Y, ITERS, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    off = tl.arange(0, BLOCK)
    for i in range(ITERS):
        idx = (i * npg + pid) * BLOCK + off
        v = tl.load(X + idx)
        tl.store(Y + idx, _trunc_plain(v).to(Y.dtype.element_ty))


res["device"] = {}
N16 = 4096 * 4096
for dtname, dt in (("fp16", torch.float16), ("fp32", torch.float32)):
    xb = torch.randn(N16, device="npu:0", dtype=dt)
    ob = torch.empty_like(xb)
    row = {}
    for blk, name in ((4096, "masked_seed_b4096"), (4096, "nomask_seed_b4096"),
                      (4096, "nomask_plain_b4096"), (8192, "nomask_plain_b8192"),
                      (16384, "nomask_plain_b16384"), (8192, "nomask_trunc_b8192")):
        g = (N16 // blk,)
        try:
            if name.startswith("masked"):
                f = lambda: k_masked_seed[g](xb, ob, N16, BLOCK=blk)
            elif name.startswith("nomask_seed"):
                f = lambda: k_nomask_seed[g](xb, ob, BLOCK=blk)
            elif name.startswith("nomask_plain"):
                f = lambda: k_nomask_plain[g](xb, ob, BLOCK=blk)
            else:
                f = lambda: k_nomask_trunc[g](xb, ob, BLOCK=blk)
            row[name] = bench(f, iters=30)
        except Exception as e:  # noqa
            row[name] = {"err": repr(e)[:200]}
    for blk, iters, name in ((8192, 8, "loop_b8192_i8"), (8192, 16, "loop_b8192_i16"),
                             (4096, 8, "loop_b4096_i8")):
        npg = N16 // (blk * iters)
        try:
            row[name] = bench(lambda: k_loop_plain[(npg,)](xb, ob, iters, BLOCK=blk), iters=30)
            row[name]["npg"] = npg
        except Exception as e:  # noqa
            row[name] = {"err": repr(e)[:200]}
    vals = [0.0, -0.0, float("nan"), float("inf"), -float("inf"), -1.0, 1.0,
            -0.5, 0.5, -1.5, 2.7, -2.7, 1024.5, -0.0001]
    xs = torch.randn(8192, device="npu:0", dtype=dt)
    xs[:14] = torch.tensor(vals, dtype=torch.float32, device="npu:0").to(dt)
    exp = torch.fix(xs)
    chk = {}
    for nm, kk in (("plain", k_nomask_plain), ("trunc", k_nomask_trunc), ("seed", k_nomask_seed)):
        os_ = torch.empty_like(xs)
        try:
            kk[(1,)](xs, os_, BLOCK=8192)
            diff = (os_ != exp) & ~(torch.isnan(os_) & torch.isnan(exp))
            chk[nm] = {"mismatch": int(diff.sum()),
                       "vals": [float(v) for v in os_[:14].tolist()]}
        except Exception as e:  # noqa
            chk[nm] = {"err": repr(e)[:200]}
    row["check"] = chk
    row["ref"] = bench(lambda: torch.fix(xb), iters=30)
    res["device"][dtname] = row

os.makedirs(ARTS, exist_ok=True)
with open(os.path.join(ARTS, "probe2.json"), "w") as fh:
    json.dump(res, fh, indent=1)
print("ok")
