"""Bounded probe 2b: direct-launch ABI/cost (must-have) + a few large-case variants.

Every config is isolated in try/except and results are flushed incrementally.
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
OUT = {"sections": {}}


def flush():
    with open(os.path.join(ART, "probe2b.json"), "w") as fh:
        json.dump(OUT, fh, indent=1, default=str)


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


def bench_us(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
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
def k_stream(A, B, alpha, n, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for start in range(tl.program_id(0) * BLOCK, n, step):
        off = start + lane
        a = tl.load(A + off, off < n, other=0.0)
        b = tl.load(B + off, off < n, other=0.0)
        tl.store(A + off, (a.to(tl.float32) - alpha * b.to(tl.float32)).to(A.dtype.element_ty),
                 off < n)


def safe(name, fn):
    try:
        return fn()
    except Exception:
        OUT["sections"].setdefault("errors", {})[name] = traceback.format_exc()[-1200:]
        flush()
        return None


# ------------------------------------------------ A: direct launcher
def section_direct():
    sec = {}
    A = torch.randn(1024, dtype=torch.float16, device="npu:0")
    B = torch.randn(1024, dtype=torch.float16, device="npu:0")
    A.flatten()[:7] = torch.tensor([0., -0., float("nan"), float("inf"), -float("inf"), -1., 1.],
                                   dtype=torch.float16, device="npu:0")
    n = A.numel()
    ck = k_masked.warmup(A, B, 0.5, n, BLOCK=4096, grid=(1,))
    launcher = ck.run
    stream = stream_handle(dev)
    gx = (n + 4095) // 4096

    def direct(Aa, Bb, alpha, nn, gxx, ckk=ck, st=stream):
        ckk.run.launch(gxx, 1, 1, st, ckk.function, ckk.packed_metadata,
                       None, None, None, Aa.data_ptr(), Bb.data_ptr(), alpha, nn, 4096)

    exp = A.clone().sub_(B, alpha=0.5)
    direct(A, B, 0.5, n, gx)
    torch.npu.synchronize()
    sec["first_call_match"] = bool(torch.equal(torch.nan_to_num(exp.float()),
                                               torch.nan_to_num(A.float())))
    A2 = torch.randn(1024, dtype=torch.float16, device="npu:0")
    B2 = torch.randn(1024, dtype=torch.float16, device="npu:0")
    exp2 = A2.clone().sub_(B2, alpha=0.25)
    direct(A2, B2, 0.25, 1024, 1)
    torch.npu.synchronize()
    sec["second_call_new_ptr_alpha_match"] = bool(torch.equal(exp2, A2))
    del A2, B2
    A3 = torch.randn((4096, 4096), dtype=torch.float16, device="npu:0")
    B3 = torch.randn((4096, 4096), dtype=torch.float16, device="npu:0")
    n3 = A3.numel()
    exp3 = A3.clone().sub_(B3, alpha=0.5)
    direct(A3, B3, 0.5, n3, n3 // 4096)
    torch.npu.synchronize()
    sec["large_same_entry_match"] = bool(torch.equal(exp3, A3))
    sec["direct_overhead_us"] = bench_us(lambda: direct(A, B, 0.5, n, gx), iters=300)
    sec["direct_inline_us"] = bench_us(
        lambda: ck.run.launch(gx, 1, 1, stream, ck.function, ck.packed_metadata,
                              None, None, None, A.data_ptr(), B.data_ptr(), 0.5, n, 4096),
        iters=300)
    sec["stream_query_us"] = bench_us(lambda: stream_handle(dev), iters=300)
    sec["plain_jit_us"] = bench_us(lambda: k_masked[(gx,)](A, B, 0.5, n, BLOCK=4096), iters=300)
    sec["torch_ref_us"] = bench_us(lambda: A.sub_(B, alpha=0.5), iters=300)
    del A3, B3, exp3
    return sec


OUT["sections"]["direct"] = safe("direct", section_direct) or {}
flush()

# ------------------------------------------------ B: large-case variants
def section_large(tag, dt, cfgs):
    A = torch.randn((4096, 4096), dtype=dt, device="npu:0")
    B = torch.randn((4096, 4096), dtype=dt, device="npu:0")
    n = A.numel()
    nbytes = 3 * n * A.element_size()
    res = {"torch_ref": bench_us(lambda: A.sub_(B, alpha=0.5))}

    def ref_val():
        keep = A.clone()
        r = keep.sub_(B, alpha=0.5)
        return r

    for kind, blk, gx in cfgs:
        key = "%s_b%d_g%d" % (kind, blk, gx)
        def go(kind=kind, blk=blk, gx=gx):
            if kind == "masked":
                k_masked[((n + blk - 1) // blk,)](A, B, 0.5, n, BLOCK=blk)
            elif kind == "even":
                k_even[(n // blk,)](A, B, 0.5, BLOCK=blk)
            elif kind == "stream":
                k_stream[(gx,)](A, B, 0.5, n, BLOCK=blk)
        if safe(tag + key, go) is None:
            continue
        torch.npu.synchronize()
        ok = safe(tag + key + "_check",
                  lambda: bool(torch.equal(ref_val(), (go(), A)[1])))
        res[key + "_us"] = bench_us(go)
        res[key + "_ok"] = ok
        flush()
    res["GBps"] = {k: nbytes / (v * 1e3) for k, v in res.items()
                   if k.endswith("_us") and isinstance(v, float)}
    res["_bytes"] = nbytes
    del A, B
    return res


CFGS = [
    ("masked", 8192, 0),
    ("masked", 16384, 0),
    ("even", 8192, 0),
    ("even", 16384, 0),
    ("stream", 4096, 40),
    ("stream", 4096, 80),
    ("stream", 2048, 80),
]
for tag, dt in (("f16", torch.float16), ("f32", torch.float32)):
    OUT["sections"][tag] = safe(tag, lambda dt=dt, tag=tag: section_large(tag, dt, CFGS)) or {}
    flush()

flush()
print(json.dumps(OUT, indent=1, default=str))
