"""Second bounded diagnostic: widen the BLOCK_M/BLOCK_N/num_warps screen for the
two bandwidth-bound shapes, verify numerics of the winners, and confirm the
small shapes do not regress (evaluator-like per-call timing).

Diagnostic only: no authoritative measurement.
"""
import json
import os
import time

import torch
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = os.path.join(ART, "probe2.json")
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


def bench_batch(fn, inner=20, iters=15, warmup=5):
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
def k_persist(
    inp, out,
    s_it, s_ik, s_ih, s_ot, s_oh, scale,
    TOPK: tl.constexpr, NT: tl.constexpr, HD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, NTILES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = on < HD
    ar = tl.arange(0, BLOCK_M)
    for t0 in range(pid_m * BLOCK_M, NT, NTILES * BLOCK_M):
        om = t0 + ar
        mask = (om[:, None] < NT) & nmask[None, :]
        base = om[:, None] * s_it + on[None, :] * s_ih
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, TOPK):
            acc += tl.load(inp + base + k * s_ik, mask=mask, other=0.0).to(tl.float32)
        acc *= scale
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


def run_persist(x, o, bm, bn, w, ntiles, scale=2.5):
    nt, topk, hd = x.shape
    k_persist[(ntiles, triton.cdiv(hd, bn))](
        x, o, *x.stride(), *o.stride(), scale,
        TOPK=topk, NT=nt, HD=hd, BLOCK_M=bm, BLOCK_N=bn, NTILES=ntiles,
        num_warps=w, num_stages=1)
    return o


CFGS = [
    (4, 1024, 4), (8, 1024, 4), (8, 1024, 8), (16, 1024, 8),
    (4, 2048, 4), (8, 2048, 8), (4, 512, 4), (8, 512, 4),
    (16, 512, 8), (2, 2048, 4), (4, 1024, 8), (8, 1024, 2),
]


def main():
    RES["torch"] = torch.__version__
    RES["triton"] = triton.__version__

    screen = {}
    for nt, inner in ((4096, 10), (512, 20), (64, 20), (8, 20)):
        x, o = make(nt)
        rows = {}
        for (bm, bn, w) in CFGS:
            key = f"bm{bm}_bn{bn}_w{w}"
            try:
                rows[key] = round(bench_batch(lambda: run_tile(x, o, bm, bn, w), inner) * 1e3, 2)
            except Exception as exc:
                rows[key] = repr(exc)[:110]
            screen[f"nt{nt}_amort"] = rows
            RES["screen"] = screen
            flush()
        for (bm, bn, w, t) in ((4, 1024, 4, 16), (8, 1024, 8, 16),
                               (4, 1024, 4, 24), (8, 1024, 8, 24)):
            key = f"persist_bm{bm}_bn{bn}_w{w}_t{t}"
            try:
                rows[key] = round(bench_batch(lambda: run_persist(x, o, bm, bn, w, t), inner) * 1e3, 2)
            except Exception as exc:
                rows[key] = repr(exc)[:110]
        screen[f"nt{nt}_amort"] = rows
        RES["screen"] = screen
        flush()
        del x, o

    # ---- numerics of the leading shapes vs the reference ----
    num = {}
    for nt, hd, topk in ((7, 128, 2), (83, 512, 4), (3, 2048, 8), (300, 7168, 8)):
        x = torch.randn(nt, topk, hd, dtype=torch.bfloat16, device="npu")
        o = torch.zeros(nt, hd, dtype=torch.bfloat16, device="npu")
        ref = x.float().sum(dim=1).mul(2.5).to(x.dtype)
        for (bm, bn, w) in ((4, 1024, 4), (8, 1024, 4)):
            bn_c = min(bn, triton.next_power_of_2(hd))
            run_tile(x, o, bm, bn_c, w)
            num[f"nt{nt}_hd{hd}_bm{bm}_bn{bn_c}"] = {
                "max_abs": float((o.float() - ref.float()).abs().max()),
                "allclose": bool(torch.allclose(o.float(), ref.float(), atol=0.015, rtol=0.015)),
                "equal": bool(torch.equal(o, ref)),
            }
        num[f"ref_nt{nt}_hd{hd}"] = "computed"
        RES["numerics"] = num
        flush()
        del x, o, ref

    # ---- evaluator-like per-call latency for the small shapes ----
    per = {}
    for nt in (1, 8, 64, 512):
        x, o = make(nt)
        ref_chain = bench_call(
            lambda: x.float().sum(dim=1).mul(2.5).to(x.dtype))
        per[f"nt{nt}_reference"] = round(ref_chain * 1e3, 2)
        for (bm, bn, w) in ((2, 512, 8), (4, 1024, 4), (1, 1024, 2)):
            try:
                per[f"nt{nt}_bm{bm}_bn{bn}_w{w}"] = round(
                    bench_call(lambda: run_tile(x, o, bm, bn, w)) * 1e3, 2)
            except Exception as exc:
                per[f"nt{nt}_bm{bm}_bn{bn}_w{w}"] = repr(exc)[:110]
        RES["percall"] = per
        flush()
        del x, o

    RES["done"] = True
    flush()
    print(json.dumps(RES, indent=1, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
