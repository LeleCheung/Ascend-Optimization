"""Target-side diagnosis #3: launcher source, minimal python path, tile sweep."""

import json
import os
import re
import sys
import time

import torch
import torch_npu
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
R = {"host": {}, "src": {}, "sweep": {}}


def save():
    path = os.path.join(ART, "probe3.json")
    with open(path, "w") as fh:
        json.dump(R, fh, indent=1, default=str)
    print("WROTE", path)


sys.path.insert(0, "tmp")
sys.path.insert(0, ".")
import main as cand  # noqa: E402

dev = torch.npu.current_device()


def batch_mean(fn, iters=300, warm=30):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


# ---------------- A. launcher source ----------------
try:
    import triton.spec.ascend.runtime.jit as ajit
    src = open(ajit.__file__).read()
    R["src"]["path"] = ajit.__file__
    R["src"]["len"] = len(src)
    keep = []
    for m in re.finditer(r"^(class |    def |def )", src, re.M):
        pass
    # keep NPULauncher class + module-level launch helpers
    idx = src.find("class NPULauncher")
    if idx >= 0:
        keep.append(src[idx:idx + 6000])
    for name in ("def make_launcher", "def _launch", "def launch", "def _arch"):
        j = src.find(name)
        if j >= 0:
            keep.append("\n#@@ " + name + "\n" + src[j:j + 3000])
    with open(os.path.join(ART, "jit_src.txt"), "w") as fh:
        fh.write("\n\n".join(keep))
except Exception as exc:  # noqa: BLE001
    R["src"]["err"] = repr(exc)

try:
    import triton.spec.ascend.runtime as art_mod
    mods = []
    for name in dir(art_mod):
        if "launch" in name.lower() or "driver" in name.lower():
            mods.append(name)
    R["src"]["art_runtime_names"] = mods
except Exception as exc:  # noqa: BLE001
    R["src"]["art_err"] = repr(exc)

# ---------------- B. minimal python fast path ----------------
g = torch.Generator().manual_seed(1729)
a32 = (torch.randn(1024, generator=g).abs() + 0.05).to(torch.float32).npu()

cand.run(a32)
launch = next(iter(cand._raw_cache.values()))
ptr = a32.data_ptr()
getter = torch_npu._C._npu_getCurrentRawStreamNoWait
device = a32.device.index

R["host"]["cand_run"] = batch_mean(lambda: cand.run(a32))
R["host"]["launch_only"] = batch_mean(lambda: launch(a32, ptr))
R["host"]["torch_log10"] = batch_mean(lambda: torch.log10_(a32))

_fast = {}


def run_min(A):
    n = A.numel()
    e = _fast.get(n)
    if e is not None and A.dtype is e[0] and A.is_contiguous():
        e[1](A.data_ptr())
    else:
        _fast[n] = (A.dtype, cand._raw_launch_entry(A, n, A.data_ptr() % 16 == 0))
        _fast[n][1](A.data_ptr())
    return A


_fast2 = {}


def run_min2(A):
    """No contiguity check in the hot path (keyed on n+dtype+contig of build)."""
    e = _fast2.get(A.numel())
    if e is not None and A.dtype is e[0]:
        e[1](A.data_ptr())
    else:
        n = A.numel()
        e = (A.dtype, cand._raw_launch_entry(A, n, A.data_ptr() % 16 == 0))
        _fast2[n] = e
        e[1](A.data_ptr())
    return A


run_min(a32)
run_min2(a32)
R["host"]["run_min"] = batch_mean(lambda: run_min(a32))
R["host"]["run_min2"] = batch_mean(lambda: run_min2(a32))
R["host"]["data_ptr_only"] = batch_mean(lambda: a32.data_ptr(), iters=2000)
save()

# ---------------- C. tile sweep on 4096x4096 ----------------


@triton.jit
def _sl(X, BLOCK: tl.constexpr):
    """straight-line single tile per program (seed shape)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.log(x.to(tl.float32)) * 0.4342944819032518
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def _sl_hint(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    x = tl.load(X + offs)
    y = tl.log(x.to(tl.float32)) * 0.4342944819032518
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def _loop(X, N, BLOCK: tl.constexpr):
    """grid-stride, main tiles unmasked, one tail."""
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


@triton.jit
def _chunk(X, N, NCHUNK, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * NCHUNK
    end = start + NCHUNK
    s = start
    while s + BLOCK <= end:
        offs = s + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        y = tl.log(x.to(tl.float32)) * 0.4342944819032518
        tl.store(X + offs, y.to(X.dtype.element_ty))
        s += BLOCK
    if s < N:
        offs = s + tl.arange(0, BLOCK)
        m = offs < N
        x = tl.load(X + offs, mask=m)
        y = tl.log(x.to(tl.float32)) * 0.4342944819032518
        tl.store(X + offs, y.to(X.dtype.element_ty), mask=m)


N = 4096 * 4096
PLAN = {
    "float16": {"sl": [8192, 16384, 32768], "loop": [1024, 2048, 4096], "chunk": [1024, 2048, 4096]},
    "float32": {"sl": [4096, 8192, 16384], "loop": [512, 1024, 2048], "chunk": [512, 1024, 2048]},
}
for dt, plan in PLAN.items():
    a = (torch.randn(N, generator=g).abs() + 0.05).to(getattr(torch, dt)).npu()
    ent = {"seed_run": batch_mean(lambda: cand.run(a), iters=80),
           "ref": batch_mean(lambda: torch.log10_(a), iters=80),
           "sl": {}, "sl_hint": {}, "loop": {}, "chunk": {}}
    for b in plan["sl"]:
        try:
            ent["sl"]["b%d" % b] = batch_mean(lambda b=b: _sl[(N // b,)](a, BLOCK=b), iters=80)
            ent["sl_hint"]["b%d" % b] = batch_mean(
                lambda b=b: _sl_hint[(N // b,)](a, BLOCK=b), iters=80)
        except Exception as exc:  # noqa: BLE001
            ent["sl"]["b%d" % b] = "ERR %s" % (str(exc)[:160],)
    for gp in (20, 40, 80):
        for b in plan["loop"]:
            tag = "g%d_b%d" % (gp, b)
            try:
                ent["loop"][tag] = batch_mean(lambda b=b, gp=gp: _loop[(gp,)](a, N, BLOCK=b), iters=80)
            except Exception as exc:  # noqa: BLE001
                ent["loop"][tag] = "ERR %s" % (str(exc)[:120],)
            try:
                nc = -(-N // gp)
                ent["chunk"][tag] = batch_mean(
                    lambda b=b, gp=gp, nc=nc: _chunk[(gp,)](a, N, nc, BLOCK=b), iters=80)
            except Exception as exc:  # noqa: BLE001
                ent["chunk"][tag] = "ERR %s" % (str(exc)[:120],)
    R["sweep"][dt] = ent
    save()

print("DONE")
