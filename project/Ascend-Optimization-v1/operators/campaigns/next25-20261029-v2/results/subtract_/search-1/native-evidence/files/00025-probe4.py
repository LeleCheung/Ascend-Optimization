"""Bounded probe 4: host-path micro-breakdown + large-case tile screening.

Every measurement is flushed immediately so a later failure cannot lose it.
"""
import json
import os
import sys
import time
import traceback

import torch
import torch_npu
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import main as cand  # noqa: E402

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = {}
dev = torch.npu.current_device()


def flush():
    with open(os.path.join(ART, "probe4.json"), "w") as fh:
        json.dump(OUT, fh, indent=1, default=str)


def stream_handle(device):
    return torch_npu._C._npu_getCurrentRawStreamNoWait(device)


def bench_us(fn, iters=300, warmup=30, sync_each=False):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    if sync_each:
        for _ in range(iters):
            fn()
            torch.npu.synchronize()
    else:
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


@triton.jit
def k_masked(A, B, alpha, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(A + offs, mask=m, other=0.0)
    b = tl.load(B + offs, mask=m, other=0.0)
    tl.store(A + offs, (a.to(tl.float32) - alpha * b.to(tl.float32)).to(A.dtype.element_ty), mask=m)


@triton.jit
def k_even(A, B, alpha, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + offs)
    b = tl.load(B + offs)
    tl.store(A + offs, (a.to(tl.float32) - alpha * b.to(tl.float32)).to(A.dtype.element_ty))


@triton.jit
def k_two(A, B, ALPHA: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + offs)
    b = tl.load(B + offs)
    tl.store(A + offs, (a.to(tl.float32) - ALPHA * b.to(tl.float32)).to(A.dtype.element_ty))


@triton.jit
def k_stream(A, B, alpha, n, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for start in range(tl.program_id(0) * BLOCK, n, step):
        off = start + lane
        a = tl.load(A + off, off < n, other=0.0)
        b = tl.load(B + off, off < n, other=0.0)
        tl.store(A + off, (a.to(tl.float32) - alpha * b.to(tl.float32)).to(A.dtype.element_ty),
                 off < n)


def safe(key, fn):
    try:
        return fn()
    except Exception:
        OUT.setdefault("errors", {})[key] = traceback.format_exc()[-800:]
        flush()
        return None


# ---------------------------------------------------------------- host path
A = torch.randn(1024, dtype=torch.float16, device="npu:0")
B = torch.randn(1024, dtype=torch.float16, device="npu:0")
n = A.numel()
st = stream_handle(dev)
host = {}
OUT["host"] = host
ck = k_masked.warmup(A, B, 0.5, n, BLOCK=4096, grid=(1,))
l5 = ck.run.launch
gx = (n + 4095) // 4096
pa, pb = A.data_ptr(), B.data_ptr()


def launch5():
    l5(gx, 1, 1, st, ck.function, ck.packed_metadata, None, None, None,
       pa, pb, 0.5, n, 4096)


host["launch5_us"] = bench_us(launch5)
flush()
host["launch5_sync_each_us"] = bench_us(launch5, iters=200, sync_each=True)
flush()


def py_prelude():
    nn = A.numel()
    d = A.device.index
    e = cand._ENTRIES.get((A.dtype, nn, d))
    ca = A.is_contiguous()
    cb = B.is_contiguous()
    eq = A.shape == B.shape
    qa = A.data_ptr()
    qb = B.data_ptr()
    al = (qa | qb) & 15
    return e, ca, cb, eq, al, stream_handle(d)


host["python_prelude_us"] = bench_us(py_prelude)
flush()
for name, fn in (("numel", lambda: A.numel()),
                 ("device_index", lambda: A.device.index),
                 ("is_contig_x2", lambda: (A.is_contiguous(), B.is_contiguous())),
                 ("shape_eq", lambda: A.shape == B.shape),
                 ("data_ptr_x2", lambda: (A.data_ptr(), B.data_ptr())),
                 ("dict_tuple_key", lambda: cand._ENTRIES.get((A.dtype, 1024, 0))),
                 ("stream_only", lambda: stream_handle(dev)),
                 ("sync_only", lambda: torch.npu.synchronize()),
                 ("float_cast", lambda: float(0.5)),
                 ("noop", lambda: None)):
    host[name + "_us"] = bench_us(fn, iters=400)
    flush()
host["cand_run_f16_us"] = bench_us(lambda: cand.run(A, B, alpha=0.5))
host["cand_run_f16_sync_us"] = bench_us(lambda: cand.run(A, B, alpha=0.5), iters=200,
                                        sync_each=True)
host["torch_ref_f16_us"] = bench_us(lambda: A.sub_(B, alpha=0.5))
host["torch_ref_f16_sync_us"] = bench_us(lambda: A.sub_(B, alpha=0.5), iters=200,
                                         sync_each=True)
flush()

# two-argument kernel: pointers only, alpha+block constexpr
two = safe("two_warmup", lambda: k_two.warmup(A, B, 0.5, BLOCK=4096, grid=(1,)))
if two is not None:
    l2 = two.run.launch
    def launch2():
        l2(gx, 1, 1, st, two.function, two.packed_metadata, None, None, None,
           pa, pb, None, None)
    host["launch2_us"] = bench_us(launch2)
    flush()
    exp = A.clone().sub_(B, alpha=0.5)
    safe("two_launch", launch2)
    torch.npu.synchronize()
    host["launch2_match"] = bool(torch.equal(exp, A))
    flush()

# ---------------------------------------------------------- large-case screen
LARGE = [(4096, torch.float16), (4096, torch.float32)]
for _sz, dt in LARGE:
    tag = str(dt).split(".")[-1]
    sec = {}
    OUT["large_" + tag] = sec
    A = torch.randn((_sz, _sz), dtype=dt, device="npu:0")
    B = torch.randn((_sz, _sz), dtype=dt, device="npu:0")
    nn = A.numel()
    nbytes = 3 * nn * A.element_size()
    sec["_bytes"] = nbytes
    sec["_n"] = nn
    sec["torch_ref_us"] = bench_us(lambda: A.sub_(B, alpha=0.5), iters=40)
    flush()

    def check(kind, blk, grid=0):
        keep = A.clone()
        ref = keep.sub_(B, alpha=0.5)
        if kind == "masked":
            k_masked[((nn + blk - 1) // blk,)](A, B, 0.5, nn, BLOCK=blk)
        elif kind == "even":
            k_even[(nn // blk,)](A, B, 0.5, BLOCK=blk)
        else:
            k_stream[(grid,)](A, B, 0.5, nn, BLOCK=blk)
        torch.npu.synchronize()
        return bool(torch.equal(ref, A))

    for kind, blk, grid in (("masked", 4096, 0), ("masked", 8192, 0), ("masked", 16384, 0),
                            ("even", 4096, 0), ("even", 8192, 0), ("even", 16384, 0),
                            ("stream", 2048, 40), ("stream", 2048, 80),
                            ("stream", 4096, 40)):
        key = "%s_b%d_g%d" % (kind, blk, grid)
        ok = safe(tag + key + "_check", lambda kind=kind, blk=blk, grid=grid: check(kind, blk, grid))
        if ok is None:
            continue
        def go(kind=kind, blk=blk, grid=grid):
            if kind == "masked":
                k_masked[((nn + blk - 1) // blk,)](A, B, 0.5, nn, BLOCK=blk)
            elif kind == "even":
                k_even[(nn // blk,)](A, B, 0.5, BLOCK=blk)
            else:
                k_stream[(grid,)](A, B, 0.5, nn, BLOCK=blk)
        us = bench_us(go, iters=40)
        sec[key + "_us"] = us
        sec[key + "_GBps"] = nbytes / (us * 1e3)
        sec[key + "_ok"] = ok
        flush()
    del A, B
    flush()

print(json.dumps(OUT, indent=1, default=str))
