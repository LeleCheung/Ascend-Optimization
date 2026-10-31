"""Bounded probe 2: direct-launch ABI/cost + large-case tile/grid variants."""
import json
import os
import sys
import time

import torch
import torch_npu
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import main as cand  # noqa: E402  tmp/main.py

OUT = {}
dev = torch.npu.current_device()
props = triton.runtime.driver.active.utils.get_device_properties(dev)
OUT["num_vectorcore"] = props.get("num_vectorcore")
OUT["num_aicore"] = props.get("num_aicore")


def stream_handle(device):
    raw = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
    if raw is not None:
        try:
            return raw(device)
        except TypeError:
            return raw()
    return torch.npu.current_stream(device).npu_stream


def bench_us(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


# ---------------------------------------------------------------- variants
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
def k_even_native(A, B, alpha, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + offs)
    b = tl.load(B + offs)
    tl.store(A + offs, a - alpha * b)


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


@triton.jit
def k_stream_even(A, B, alpha, n, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for start in range(tl.program_id(0) * BLOCK, n, step):
        off = start + lane
        a = tl.load(A + off)
        b = tl.load(B + off)
        tl.store(A + off, (a.to(tl.float32) - alpha * b.to(tl.float32)).to(A.dtype.element_ty))


def run_variant(kind, A, B, n, block, grid=None):
    if kind == "masked":
        k_masked[((n + block - 1) // block,)](A, B, 0.5, n, BLOCK=block)
    elif kind == "even":
        k_even[(n // block,)](A, B, 0.5, BLOCK=block)
    elif kind == "even_native":
        k_even_native[(n // block,)](A, B, 0.5, BLOCK=block)
    elif kind == "stream":
        k_stream[(grid,)](A, B, 0.5, n, BLOCK=block)
    elif kind == "stream_even":
        k_stream_even[(grid,)](A, B, 0.5, n, BLOCK=block)
    else:
        raise ValueError(kind)


# ------------------------------------------------- direct launcher ABI test
A, B = (torch.randn(1024, dtype=torch.float16, device="npu:0"),
        torch.randn(1024, dtype=torch.float16, device="npu:0"))
A.flatten()[:7] = torch.tensor([0., -0., float("nan"), float("inf"), -float("inf"), -1., 1.],
                              dtype=torch.float16, device="npu:0")
n = A.numel()
ck = k_masked.warmup(A, B, 0.5, n, BLOCK=4096, grid=(1,))
launcher = ck.run
stream = stream_handle(dev)
gridx = (n + 4096 - 1) // 4096


def direct(Aa, Bb, alpha, nn, blk, gx):
    launcher.launch(gx, 1, 1, stream, ck.function, ck.packed_metadata,
                    None, None, None, Aa.data_ptr(), Bb.data_ptr(), alpha, nn, blk)


d = {}
try:
    exp = A.clone().sub_(B, alpha=0.5)
    direct(A, B, 0.5, n, 4096, gridx)
    torch.npu.synchronize()
    d["first_call_match"] = bool(torch.equal(torch.nan_to_num(exp.float()),
                                             torch.nan_to_num(A.float())))
    A2, B2 = (torch.randn(1024, dtype=torch.float16, device="npu:0"),
              torch.randn(1024, dtype=torch.float16, device="npu:0"))
    exp2 = A2.clone().sub_(B2, alpha=0.25)
    direct(A2, B2, 0.25, 1024, 4096, 1)
    torch.npu.synchronize()
    d["second_call_new_ptr_alpha_match"] = bool(torch.equal(exp2, A2))
    # big tile through the same entry: gridX larger, n as runtime arg
    A3 = torch.randn((4096, 4096), dtype=torch.float16, device="npu:0")
    B3 = torch.randn((4096, 4096), dtype=torch.float16, device="npu:0")
    n3 = A3.numel()
    exp3 = A3.clone().sub_(B3, alpha=0.5)
    direct(A3, B3, 0.5, n3, 4096, n3 // 4096)
    torch.npu.synchronize()
    d["large_match"] = bool(torch.equal(exp3, A3))
    d["overhead_us"] = bench_us(lambda: direct(A, B, 0.5, n, 4096, gridx), iters=300)
    d["overhead_us_cached_stream"] = bench_us(
        lambda: launcher.launch(gridx, 1, 1, stream, ck.function, ck.packed_metadata,
                                None, None, None, A.data_ptr(), B.data_ptr(), 0.5, n, 4096),
        iters=300)
    d["plain_jit_us"] = bench_us(lambda: k_masked[(gridx,)](A, B, 0.5, n, BLOCK=4096), iters=300)
    d["torch_ref_us"] = bench_us(lambda: A.sub_(B, alpha=0.5), iters=300)
    del A3, B3, exp3
except Exception as exc:
    d["error"] = repr(exc)
OUT["direct_launch"] = d

# ------------------------------------------------------ large-case variants
big = {}
for tag, dt in (("f16", torch.float16), ("bf16", torch.bfloat16), ("f32", torch.float32)):
    A = torch.randn((4096, 4096), dtype=dt, device="npu:0")
    B = torch.randn((4096, 4096), dtype=dt, device="npu:0")
    nn = A.numel()
    nbytes = 3 * nn * A.element_size()
    res = {}
    res["torch_ref"] = bench_us(lambda: A.sub_(B, alpha=0.5))
    for blk in (4096, 8192, 16384, 32768):
        try:
            res["masked_%d" % blk] = bench_us(
                lambda blk=blk: run_variant("masked", A, B, nn, blk))
            res["even_%d" % blk] = bench_us(
                lambda blk=blk: run_variant("even", A, B, nn, blk))
        except Exception as exc:
            res["masked_%d" % blk] = "ERR %r" % (exc,)
    for gx in (40, 80, 160):
        for blk in (4096, 8192, 16384):
            res["stream_g%d_b%d" % (gx, blk)] = bench_us(
                lambda gx=gx, blk=blk: run_variant("stream", A, B, nn, blk, gx))
            res["stream_even_g%d_b%d" % (gx, blk)] = bench_us(
                lambda gx=gx, blk=blk: run_variant("stream_even", A, B, nn, blk, gx))
    if dt is not torch.float32:
        for blk in (8192, 16384):
            res["even_native_%d" % blk] = bench_us(
                lambda blk=blk: run_variant("even_native", A, B, nn, blk))
    # correctness spot-check of the fastest shapes
    for kind, blk, gx in (("even", 16384, None), ("stream_even", 16384, 40)):
        ref = A.clone()
        if kind == "even":
            k_even[(nn // blk,)](A, B, 0.5, BLOCK=blk)
        else:
            k_stream_even[(gx,)](A, B, 0.5, nn, BLOCK=blk)
        torch.npu.synchronize()
        res["check_%s_%d" % (kind, blk)] = bool(torch.allclose(ref.sub_(B, alpha=0.5), A,
                                                               atol=0, rtol=0))
    big[tag] = {"us": res, "GBps": {k: (nbytes / (v * 1e3) if isinstance(v, float) else v)
                                    for k, v in res.items()}}
    del A, B
OUT["large"] = big

art = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
with open(os.path.join(art, "probe2.json"), "w") as fh:
    json.dump(OUT, fh, indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str))
