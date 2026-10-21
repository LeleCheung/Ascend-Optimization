"""Third bounded diagnostic: push BLOCK_M beyond 16 for the bandwidth-bound
shapes, test a single top-k tile load (tl.sum over axis 1) versus the unrolled
loop, verify numerics of the final dispatch, and measure per-call latency of the
exact dispatch table that will be submitted.

Diagnostic only: no authoritative measurement.
"""
import json
import os
import time

import torch
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = os.path.join(ART, "probe3.json")
RES = {}


def flush():
    with open(OUT, "w") as f:
        json.dump(RES, f, indent=1, sort_keys=True, default=str)


def bench_call(fn, iters=80, warmup=15):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def bench_batch(fn, inner=10, iters=15, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3 / inner)
    ts.sort()
    return ts[len(ts) // 2]


@triton.jit
def k_tile(
    inp, out,
    s_it, s_ik, s_ih, s_ot, s_oh, scale,
    TOPK: tl.constexpr, NT: tl.constexpr, HD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (om[:, None] < NT) & (on[None, :] < HD)
    base = om[:, None] * s_it + on[None, :] * s_ih
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, TOPK):
        acc += tl.load(inp + base + k * s_ik, mask=mask, other=0.0).to(tl.float32)
    acc *= scale
    tl.store(out + om[:, None] * s_ot + on[None, :] * s_oh, acc, mask=mask)


@triton.jit
def k_topktile(
    inp, out,
    s_it, s_ik, s_ih, s_ot, s_oh, scale,
    TOPK: tl.constexpr, NT: tl.constexpr, HD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ok = tl.arange(0, TOPK)
    mask = (om[:, None, None] < NT) & (on[None, None, :] < HD)
    ptrs = (inp + om[:, None, None] * s_it + ok[None, :, None] * s_ik
            + on[None, None, :] * s_ih)
    vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(vals, axis=1) * scale
    tl.store(out + om[:, None] * s_ot + on[None, :] * s_oh, acc, mask=mask)


def make(nt, hd=7168, topk=8):
    x = torch.randn(nt, topk, hd, dtype=torch.bfloat16, device="npu")
    o = torch.empty(nt, hd, dtype=torch.bfloat16, device="npu")
    return x, o


def run_tile(x, o, bm, bn, w, scale=2.5):
    nt, topk, hd = x.shape
    k_tile[(triton.cdiv(nt, bm), triton.cdiv(hd, bn))](
        x, o, *x.stride(), *o.stride(), scale,
        TOPK=topk, NT=nt, HD=hd, BLOCK_M=bm, BLOCK_N=bn,
        num_warps=w, num_stages=1)
    return o


def run_topktile(x, o, bm, bn, w, scale=2.5):
    nt, topk, hd = x.shape
    k_topktile[(triton.cdiv(nt, bm), triton.cdiv(hd, bn))](
        x, o, *x.stride(), *o.stride(), scale,
        TOPK=topk, NT=nt, HD=hd, BLOCK_M=bm, BLOCK_N=bn,
        num_warps=w, num_stages=1)
    return o


TILE_CFGS = [
    (16, 1024, 8), (32, 1024, 8), (64, 1024, 8), (16, 2048, 8),
    (32, 2048, 8), (16, 1024, 4), (32, 1024, 4), (16, 4096, 8),
    (64, 2048, 8), (128, 1024, 8),
]
TOP_CFGS = [(1, 1024, 4), (4, 1024, 8), (8, 1024, 8), (16, 1024, 8), (8, 512, 8)]


def main():
    RES["torch"] = torch.__version__
    RES["triton"] = triton.__version__

    screen = {}
    for nt, inner in ((4096, 10), (512, 20)):
        x, o = make(nt)
        rows = {}
        for (bm, bn, w) in TILE_CFGS:
            key = f"loop_bm{bm}_bn{bn}_w{w}"
            try:
                rows[key] = round(bench_batch(lambda: run_tile(x, o, bm, bn, w), inner) * 1e3, 2)
            except Exception as exc:
                rows[key] = repr(exc)[:110]
            screen[f"nt{nt}"] = rows
            RES["screen"] = screen
            flush()
        for (bm, bn, w) in TOP_CFGS:
            key = f"topk_bm{bm}_bn{bn}_w{w}"
            try:
                rows[key] = round(bench_batch(lambda: run_topktile(x, o, bm, bn, w), inner) * 1e3, 2)
            except Exception as exc:
                rows[key] = repr(exc)[:110]
            screen[f"nt{nt}"] = rows
            RES["screen"] = screen
            flush()
        del x, o

    # ---- numerics of the final dispatch candidates on the correctness shapes ----
    num = {}
    for nt, hd, topk in ((7, 128, 2), (83, 512, 4), (3, 2048, 8), (300, 7168, 8)):
        x = torch.randn(nt, topk, hd, dtype=torch.bfloat16, device="npu")
        ref = x.float().sum(dim=1).mul(2.5).to(x.dtype)
        for (bm, w) in ((16, 8), (32, 8)):
            bn = min(1024, triton.next_power_of_2(hd))
            o = torch.zeros(nt, hd, dtype=torch.bfloat16, device="npu")
            run_tile(x, o, bm, bn, w)
            num[f"loop_nt{nt}_hd{hd}_bm{bm}_bn{bn}"] = {
                "max_abs": float((o.float() - ref.float()).abs().max()),
                "equal": bool(torch.equal(o, ref)),
            }
        for (bm, w) in ((4, 8),):
            bn = min(1024, triton.next_power_of_2(hd))
            o2 = torch.zeros(nt, hd, dtype=torch.bfloat16, device="npu")
            run_topktile(x, o2, bm, bn, w)
            num[f"topk_nt{nt}_hd{hd}_bm{bm}_bn{bn}"] = {
                "max_abs": float((o2.float() - ref.float()).abs().max()),
                "equal": bool(torch.equal(o2, ref)),
            }
        RES["numerics"] = num
        flush()
        del x, ref

    # ---- per-call latency, evaluator-like, for the candidate dispatch table ----
    per = {}
    for nt in (1, 8, 64, 512):
        x, o = make(nt)
        per[f"nt{nt}_reference"] = round(
            bench_call(lambda: x.float().sum(dim=1).mul(2.5).to(x.dtype)) * 1e3, 2)
        for (bm, bn, w) in ((16, 1024, 8), (2, 512, 8), (4, 1024, 4), (32, 1024, 8)):
            try:
                per[f"nt{nt}_bm{bm}_bn{bn}_w{w}"] = round(
                    bench_call(lambda: run_tile(x, o, bm, bn, w)) * 1e3, 2)
            except Exception as exc:
                per[f"nt{nt}_bm{bm}_bn{bn}_w{w}"] = repr(exc)[:110]
        per[f"nt{nt}_reference2"] = round(
            bench_call(lambda: x.float().sum(dim=1).mul(2.5).to(x.dtype)) * 1e3, 2)
        RES["percall"] = per
        flush()
        del x, o

    RES["done"] = True
    flush()
    print(json.dumps(RES, indent=1, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
