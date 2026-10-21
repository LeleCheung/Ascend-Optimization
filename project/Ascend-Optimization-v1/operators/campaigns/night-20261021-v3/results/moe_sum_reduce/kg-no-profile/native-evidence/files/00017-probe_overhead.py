"""Bounded diagnostic on the target: decompose the fixed per-call cost and
screen device-side tile configs for moe_sum_reduce.

Writes results incrementally to $KGS_DEBUG_ARTIFACTS/probe.json so a partial
run still yields evidence. Diagnostic only: no authoritative measurement.
"""
import json
import os
import time

import torch
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = os.path.join(ART, "probe.json")
RES = {}


def flush():
    with open(OUT, "w") as f:
        json.dump(RES, f, indent=1, sort_keys=True)


def sync():
    torch.npu.synchronize()


def bench_call(fn, iters=60, warmup=10):
    """Per-call walltime incl. one sync, median ms -- closest to the evaluator."""
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def bench_batch(fn, inner=50, iters=15, warmup=5):
    """Amortized per-call ms over `inner` enqueues with a single sync:
    ~ host enqueue cost + device time, with the sync cost divided out."""
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1e3 / inner)
    ts.sort()
    return ts[len(ts) // 2]


@triton.jit
def k_tile(
    inp, out,
    s_it, s_ik, s_ih, s_ot, s_oh, scale,
    TOPK: tl.constexpr, NT: tl.constexpr, HD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, MASKED: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if MASKED:
        mask = (om[:, None] < NT) & (on[None, :] < HD)
    else:
        mask = on[None, :] < HD
    base = om[:, None] * s_it + on[None, :] * s_ih
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, TOPK):
        acc += tl.load(inp + base + k * s_ik, mask=mask, other=0.0).to(tl.float32)
    acc *= scale
    tl.store(out + om[:, None] * s_ot + on[None, :] * s_oh, acc, mask=mask)


@triton.jit
def k_trivial(p, a, b, c, d, e, f, g, BLOCK: tl.constexpr):
    off = tl.arange(0, BLOCK)
    tl.store(p + off, tl.zeros((BLOCK,), dtype=tl.float32), mask=off < 0)


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


def run_tile(x, o, bm, bn, warps, masked=True):
    nt, topk, hd = x.shape
    k_tile[(triton.cdiv(nt, bm), triton.cdiv(hd, bn))](
        x, o, *x.stride(), *o.stride(), 2.5,
        TOPK=topk, NT=nt, HD=hd, BLOCK_M=bm, BLOCK_N=bn, MASKED=masked,
        num_warps=warps, num_stages=1)
    return o


def run_persist(x, o, bm, bn, warps, ntiles):
    nt, topk, hd = x.shape
    k_persist[(ntiles, triton.cdiv(hd, bn))](
        x, o, *x.stride(), *o.stride(), 2.5,
        TOPK=topk, NT=nt, HD=hd, BLOCK_M=bm, BLOCK_N=bn, NTILES=ntiles,
        num_warps=warps, num_stages=1)
    return o


def main():
    RES["torch"] = torch.__version__
    RES["triton"] = triton.__version__
    try:
        import torch_npu
        RES["torch_npu"] = torch_npu.__version__
    except Exception as exc:  # pragma: no cover
        RES["torch_npu"] = repr(exc)

    # ---------- Phase 1: per-call cost decomposition on the smallest shape ----------
    x1, o1 = make(1)
    big_x, big_o = make(4096)

    def ref_chain():
        return x1.float().sum(dim=1).mul(2.5).to(x1.dtype)

    base = {}
    base["sync_only"] = bench_call(lambda: None)
    base["empty_alloc"] = bench_call(
        lambda: torch.empty((1, 7168), dtype=torch.bfloat16, device="npu"))
    base["reference_chain"] = bench_call(ref_chain)
    base["seed_run"] = bench_call(lambda: run_tile(x1, o1, 2, 512, 8))
    base["launch_only_many_args"] = bench_call(lambda: run_tile(x1, o1, 2, 512, 8))
    base["trivial_kernel_8args"] = bench_call(
        lambda: k_trivial[(14,)](o1, 1, 2, 3, 4, 5, 6, 7, BLOCK=512, num_warps=8))
    base["stride_calls"] = bench_call(lambda: (x1.stride(), o1.stride(), x1.shape))
    RES["phase1_call_us"] = {k: round(v * 1e3, 2) for k, v in base.items()}
    flush()

    # amortized (sync cost divided out): host enqueue + device
    amb = {}
    amb["reference_chain"] = bench_batch(ref_chain)
    amb["seed_run"] = bench_batch(lambda: run_tile(x1, o1, 2, 512, 8))
    amb["trivial_kernel_8args"] = bench_batch(
        lambda: k_trivial[(14,)](o1, 1, 2, 3, 4, 5, 6, 7, BLOCK=512, num_warps=8))
    amb["empty_alloc"] = bench_batch(
        lambda: torch.empty((1, 7168), dtype=torch.bfloat16, device="npu"))
    RES["phase1_amortized_us"] = {k: round(v * 1e3, 2) for k, v in amb.items()}
    flush()

    # ---------- Phase 2: config screen, amortized device cost ----------
    cfgs = [
        (1, 1024, 1), (1, 1024, 2), (1, 2048, 2), (1, 2048, 4),
        (1, 4096, 4), (2, 512, 4), (2, 1024, 4), (4, 1024, 4),
        (1, 512, 1), (1, 512, 2), (2, 256, 4), (8, 256, 4),
    ]
    screen = {}
    for nt, inner in ((64, 30), (512, 20), (4096, 10)):
        x, o = make(nt)
        rows = {}
        for (bm, bn, w) in cfgs:
            if bn > 7168:
                continue
            masked = (nt % bm) != 0
            try:
                t = bench_batch(lambda: run_tile(x, o, bm, bn, w, masked), inner=inner)
                rows[f"bm{bm}_bn{bn}_w{w}"] = round(t * 1e3, 2)
            except Exception as exc:
                rows[f"bm{bm}_bn{bn}_w{w}"] = repr(exc)[:120]
            RES["phase2_screen"] = screen
            flush()
        # no-mask variant checks (hidden 7168 divides by 256/512/1024)
        for (bm, bn, w) in ((1, 1024, 2), (1, 512, 2), (2, 512, 4)):
            if nt % bm == 0:
                try:
                    t = bench_batch(
                        lambda: run_tile(x, o, bm, bn, w, False), inner=inner)
                    rows[f"nomask_bm{bm}_bn{bn}_w{w}"] = round(t * 1e3, 2)
                except Exception as exc:
                    rows[f"nomask_bm{bm}_bn{bn}_w{w}"] = repr(exc)[:120]
        screen[f"nt{nt}"] = rows
        RES["phase2_screen"] = screen
        flush()
        del x, o

    # ---------- Phase 3: persistent variants ----------
    pers = {}
    for nt in (512, 4096):
        x, o = make(nt)
        rows = {}
        for (bm, bn, w, ntiles) in ((1, 1024, 2, 20), (1, 1024, 2, 40),
                                    (2, 1024, 4, 20), (1, 2048, 2, 20),
                                    (4, 1024, 4, 16)):
            try:
                t = bench_batch(lambda: run_persist(x, o, bm, bn, w, ntiles), inner=10)
                rows[f"bm{bm}_bn{bn}_w{w}_t{ntiles}"] = round(t * 1e3, 2)
            except Exception as exc:
                rows[f"bm{bm}_bn{bn}_w{w}_t{ntiles}"] = repr(exc)[:120]
            pers[f"nt{nt}"] = rows
            RES["phase3_persistent"] = pers
            flush()
        del x, o

    RES["done"] = True
    flush()
    print(json.dumps(RES, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
