"""Target-side diagnosis #4: device-only time (raw launcher) for tiling/log variants."""

import json
import os
import sys
import time

import torch
import torch_npu
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
R = {"src": {}, "dev": {}}


def save():
    path = os.path.join(ART, "probe4.json")
    with open(path, "w") as fh:
        json.dump(R, fh, indent=1, default=str)
    print("WROTE", path)


sys.path.insert(0, "tmp")
sys.path.insert(0, ".")
import main as cand  # noqa: E402

dev = torch.npu.current_device()
getter = torch_npu._C._npu_getCurrentRawStreamNoWait


def batch_mean(fn, iters=80, warm=20):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


# ---- launcher source ----
p = "/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/runtime/jit.py"
try:
    src = open(p).read()
    R["src"]["path"] = p
    i = src.find("class NPULauncher")
    R["src"]["npu_launcher"] = src[i:i + 5000] if i >= 0 else "NOT FOUND"
    j = src.find("def make_launcher")
    R["src"]["make_launcher"] = src[j:j + 4000] if j >= 0 else "NOT FOUND"
    k = src.find("class NPULauncher", 0)
    R["src"]["len"] = len(src)
except Exception as exc:  # noqa: BLE001
    R["src"]["err"] = repr(exc)
save()


@triton.jit
def _ln(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.log(x.to(tl.float32)) * 0.4342944819032518
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def _ln2(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.log2(x.to(tl.float32)) * 0.30102999566398120
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def _lnat(X, BLOCK: tl.constexpr):
    """keep the operand dtype: no explicit fp32 conversion."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.log2(x) * 0.30102999566398120
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def _loop(X, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    s = pid * BLOCK
    while s + BLOCK <= N:
        offs = s + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        y = tl.log2(x.to(tl.float32)) * 0.30102999566398120
        tl.store(X + offs, y.to(X.dtype.element_ty))
        s += step
    if s < N:
        offs = s + tl.arange(0, BLOCK)
        m = offs < N
        x = tl.load(X + offs, mask=m)
        y = tl.log2(x.to(tl.float32)) * 0.30102999566398120
        tl.store(X + offs, y.to(X.dtype.element_ty), mask=m)


@triton.jit
def _chunk(X, N, NCHUNK, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * NCHUNK
    end = start + NCHUNK
    s = start
    while s + BLOCK <= end:
        offs = s + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        y = tl.log2(x.to(tl.float32)) * 0.30102999566398120
        tl.store(X + offs, y.to(X.dtype.element_ty))
        s += BLOCK
    if s < N:
        offs = s + tl.arange(0, BLOCK)
        m = offs < N
        x = tl.load(X + offs, mask=m)
        y = tl.log2(x.to(tl.float32)) * 0.30102999566398120
        tl.store(X + offs, y.to(X.dtype.element_ty), mask=m)


def raw_time(kern, tensor, extra, cexpr, gx, iters=80, **opts):
    """Device+pybind time only: compiled launcher invoked directly with a raw pointer."""
    compiled = kern.warmup(tensor, *extra, grid=(1,), **cexpr, **opts)
    launcher = compiled.run
    if (getattr(launcher, "compile_only", False)
            or getattr(launcher, "enable_msprof_register_tensor", False)):
        raise RuntimeError("bookkeeping launcher")
    call, fn, md = launcher.launch, compiled.function, compiled.packed_metadata
    ptr = tensor.data_ptr()
    params = tuple([ptr] + list(extra) + [None] * len(cexpr))

    def go():
        call(gx, 1, 1, getter(dev), fn, md, None, None, None, *params)
    return batch_mean(go, iters=iters)


N = 4096 * 4096
g = torch.Generator().manual_seed(1729)

for dt in ("float16", "float32"):
    a = (torch.randn(N, generator=g).abs() + 0.05).to(getattr(torch, dt)).npu()
    ent = {"ref": batch_mean(lambda: torch.log10_(a)),
           "seed_run": batch_mean(lambda: cand.run(a))}
    big = 32768 if dt == "float16" else 16384

    def rec(name, fn):
        try:
            ent[name] = fn()
        except Exception as exc:  # noqa: BLE001
            ent[name] = "ERR %s" % (str(exc)[:200],)
        save()

    rec("sl_ln_%d" % big, lambda: raw_time(_ln, a, (), {"BLOCK": big}, N // big))
    rec("sl_ln2_%d" % big, lambda: raw_time(_ln2, a, (), {"BLOCK": big}, N // big))
    rec("sl_lnat_%d" % big, lambda: raw_time(_lnat, a, (), {"BLOCK": big}, N // big))
    if dt == "float16":
        rec("sl_ln2_65536", lambda: raw_time(_ln2, a, (), {"BLOCK": 65536}, N // 65536))
        rec("sl_ln2_16384", lambda: raw_time(_ln2, a, (), {"BLOCK": 16384}, N // 16384))
    else:
        rec("sl_ln2_8192", lambda: raw_time(_ln2, a, (), {"BLOCK": 8192}, N // 8192))
        rec("sl_ln2_32768", lambda: raw_time(_ln2, a, (), {"BLOCK": 32768}, N // 32768))
    for gp in (20, 40, 80):
        for b in (16384, big):
            rec("loop_ns1_g%d_b%d" % (gp, b), lambda gp=gp, b=b: raw_time(
                _loop, a, (N,), {"BLOCK": b}, gp, num_stages=1))
        nc = -(-N // gp)
        rec("chunk_ns1_g%d_b%d" % (gp, big), lambda gp=gp, b=big, nc=nc: raw_time(
            _chunk, a, (N, nc), {"BLOCK": b}, gp, num_stages=1))
    rec("loop_ns2_g40_b%d" % big, lambda: raw_time(
        _loop, a, (N,), {"BLOCK": big}, 40, num_stages=2))
    R["dev"][dt] = ent
    save()

print("DONE")
