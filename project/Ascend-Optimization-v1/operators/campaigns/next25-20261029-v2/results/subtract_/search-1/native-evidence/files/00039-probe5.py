"""Bounded probe 5: large-tile screening for 2-byte and 4-byte element types."""
import json
import os
import time
import traceback

import torch
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = {}


def flush():
    with open(os.path.join(ART, "probe5.json"), "w") as fh:
        json.dump(OUT, fh, indent=1, default=str)


def bench_us(fn, iters=40, warmup=8):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


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


CFGS = {
    "float16": [("even", 16384, 0), ("even", 24576, 0), ("even", 32768, 0),
                ("stream", 8192, 40)],
    "bfloat16": [("even", 16384, 0), ("even", 32768, 0)],
    "float32": [("even", 8192, 0), ("even", 12288, 0), ("even", 16384, 0),
                ("stream", 8192, 40), ("stream", 8192, 80)],
}

for dname, cfgs in CFGS.items():
    dt = getattr(torch, dname)
    sec = {}
    OUT[dname] = sec
    A = torch.randn((4096, 4096), dtype=dt, device="npu:0")
    B = torch.randn((4096, 4096), dtype=dt, device="npu:0")
    n = A.numel()
    nbytes = 3 * n * A.element_size()
    sec["_bytes"] = nbytes
    sec["torch_ref_us"] = bench_us(lambda: A.sub_(B, alpha=0.5))
    flush()
    for kind, blk, gx in cfgs:
        key = "%s_b%d_g%d" % (kind, blk, gx)
        try:
            keep = A.clone()
            ref = keep.sub_(B, alpha=0.5)
            if kind == "even":
                k_even[(n // blk,)](A, B, 0.5, BLOCK=blk)
            else:
                k_stream[(gx,)](A, B, 0.5, n, BLOCK=blk)
            torch.npu.synchronize()
            ok = bool(torch.equal(ref, A))

            def go(kind=kind, blk=blk, gx=gx):
                if kind == "even":
                    k_even[(n // blk,)](A, B, 0.5, BLOCK=blk)
                else:
                    k_stream[(gx,)](A, B, 0.5, n, BLOCK=blk)
            us = bench_us(go)
            sec[key] = {"us": us, "GBps": nbytes / (us * 1e3), "ok": ok}
        except Exception:
            sec[key] = {"error": traceback.format_exc()[-300:]}
        flush()
    # native-dtype (no fp32 upcast) comparison at the best tile
    try:
        blk = 16384 if dt is not torch.float32 else 8192
        keep = A.clone()
        ref = keep.sub_(B, alpha=0.5)
        k_even_native[(n // blk,)](A, B, 0.5, BLOCK=blk)
        torch.npu.synchronize()
        ok = bool(torch.equal(ref, A))
        us = bench_us(lambda blk=blk: k_even_native[(n // blk,)](A, B, 0.5, BLOCK=blk))
        sec["even_native_b%d" % blk] = {"us": us, "GBps": nbytes / (us * 1e3), "ok": ok}
    except Exception:
        sec["even_native"] = {"error": traceback.format_exc()[-300:]}
    del A, B
    flush()

print(json.dumps(OUT, indent=1, default=str))
