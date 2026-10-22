"""Bounded diagnostic: identify torch.cumulative_trapezoid rounding chain and
microbenchmark contiguous-specialized scan variants on the target device.

Diagnostic evidence only; not an official measurement.
"""
import inspect
import json
import os
import time
import traceback

import torch
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
DEV = "npu:0"
OUT = {}


def sync():
    torch.npu.synchronize()


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1e3


# ---------------------------------------------------------------- part A: source
def part_a():
    try:
        import torch.functional as TF
        OUT["src"] = inspect.getsource(TF.cumulative_trapezoid)
        OUT["src_file"] = inspect.getsourcefile(TF.cumulative_trapezoid)
    except Exception:
        OUT["src_err"] = traceback.format_exc()
    OUT["torch_version"] = torch.__version__
    OUT["triton_version"] = triton.__version__


# --------------------------------------------------- part B: rounding chain probe
def part_b():
    torch.manual_seed(0)
    cases = []
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        for dx in (1, 0.25, -0.5, 1.3, 0.125, -2):
            cases.append((dt, dx))
    grid_res = {}
    for dt, dx in cases:
        y = torch.randn(4, 4097, device=DEV).to(dt)
        ref = torch.cumulative_trapezoid(y, dx=dx, dim=-1)
        p = y[..., 1:] + y[..., :-1]  # adjacent sum in input dtype
        modes = {
            "pair_dx_div2": ((p * dx) / 2),
            "pair_scalehalf": (p * (dx / 2)),
            "pair_div2_dx": ((p / 2) * dx),
        }
        entry = {}
        for name, inc in modes.items():
            incf = inc.float()
            a = incf.cumsum(-1).to(dt)
            b = incf.cumsum(-1).to(torch.float32).to(dt)
            entry[name + "|f32acc"] = float((a.float() - ref.float()).abs().max())
            entry[name + "|f16acc"] = float(
                (inc.cumsum(-1).float() - ref.float()).abs().max()
            )
            entry[name + "_exact"] = float((a.float() - ref.float()).abs().max())
        # no intermediate rounding at all
        exact = ((y.float()[..., 1:] + y.float()[..., :-1]) * (dx / 2)).cumsum(-1).to(dt)
        entry["no_round|f32acc"] = float((exact.float() - ref.float()).abs().max())
        grid_res[f"{str(dt).split('.')[-1]}|dx={dx}"] = entry
    OUT["chain_maxabs"] = grid_res


# ------------------------------------------------------- kernels for benchmarks
@triton.jit
def k_contig(y_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    m = k < (N - 1)
    yb = y_ptr + row * N
    a = tl.load(yb + k, mask=m, other=0.0).to(tl.float32)
    b = tl.load(yb + k + 1, mask=m, other=0.0).to(tl.float32)
    s = (a + b) * 0.5
    cs = tl.cumsum(s, axis=0)
    ob = out_ptr + row * (N - 1)
    tl.store(ob + k, cs, mask=m)


@triton.jit
def k_nocumsum(y_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    m = k < (N - 1)
    yb = y_ptr + row * N
    a = tl.load(yb + k, mask=m, other=0.0).to(tl.float32)
    b = tl.load(yb + k + 1, mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * (N - 1) + k, (a + b) * 0.5, mask=m)


@triton.jit
def k_loop(y_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr, NCH: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    carry = 0.0
    for c in tl.static_range(NCH):
        idx = c * BLOCK + k
        m = idx < (N - 1)
        yb = y_ptr + row * N
        a = tl.load(yb + idx, mask=m, other=0.0).to(tl.float32)
        b = tl.load(yb + idx + 1, mask=m, other=0.0).to(tl.float32)
        s = (a + b) * 0.5
        cs = tl.cumsum(s, axis=0) + carry
        tl.store(out_ptr + row * (N - 1) + idx, cs, mask=m)
        carry = tl.sum(tl.where(k == BLOCK - 1, cs, 0.0), axis=0)


@triton.jit
def k_chunksum(y_ptr, part_ptr, N: tl.constexpr, BLOCK: tl.constexpr, NCH: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // NCH
    c = pid % NCH
    k = tl.arange(0, BLOCK)
    idx = c * BLOCK + k
    m = idx < (N - 1)
    yb = y_ptr + row * N
    a = tl.load(yb + idx, mask=m, other=0.0).to(tl.float32)
    b = tl.load(yb + idx + 1, mask=m, other=0.0).to(tl.float32)
    s = (a + b) * 0.5
    tl.store(part_ptr + pid, tl.sum(s, axis=0))


@triton.jit
def k_apply(y_ptr, out_ptr, part_ptr, N: tl.constexpr, BLOCK: tl.constexpr,
            NCH: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // NCH
    c = pid % NCH
    k = tl.arange(0, BLOCK)
    idx = c * BLOCK + k
    m = idx < (N - 1)
    yb = y_ptr + row * N
    a = tl.load(yb + idx, mask=m, other=0.0).to(tl.float32)
    b = tl.load(yb + idx + 1, mask=m, other=0.0).to(tl.float32)
    s = (a + b) * 0.5
    cs = tl.cumsum(s, axis=0)
    # exclusive prefix of previous chunks within the row
    ci = tl.arange(0, NCH)
    pm = ci < c
    prev = tl.load(part_ptr + row * NCH + ci, mask=pm, other=0.0)
    carry = tl.sum(prev, axis=0)
    tl.store(out_ptr + row * (N - 1) + idx, cs + carry, mask=m)


def part_c():
    shapes = [
        ("fp16_1024x4096", torch.float16, (1024, 4096), -1),
        ("fp32_1024x4096", torch.float32, (1024, 4096), -1),
        ("fp16_128x1024", torch.float16, (128, 1024), -1),
        ("fp32_128x1024", torch.float32, (128, 1024), -1),
    ]
    res = {}
    for name, dt, shape, dim in shapes:
        y = torch.randn(shape, device=DEV).to(dt)
        ref = torch.cumulative_trapezoid(y, dx=1, dim=dim)
        rows = shape[0]
        N = shape[1]
        o = torch.empty((rows, N - 1), dtype=dt, device=DEV)
        BLOCK = triton.next_power_of_2(N - 1)
        e = {}
        e["ref_ms"] = bench(lambda: torch.cumulative_trapezoid(y, dx=1, dim=dim))
        e["k_contig_ms"] = bench(lambda: k_contig[(rows,)](y, o, N, BLOCK=BLOCK))
        e["k_contig_err"] = float((o.float() - ref.float()).abs().max())
        e["k_nocumsum_ms"] = bench(lambda: k_nocumsum[(rows,)](y, o, N, BLOCK=BLOCK))
        for bs in (512, 1024, 2048):
            if bs >= BLOCK:
                continue
            nch = triton.cdiv(N - 1, bs)
            try:
                e[f"k_loop_bs{bs}_ms"] = bench(
                    lambda: k_loop[(rows,)](y, o, N, BLOCK=bs, NCH=nch)
                )
                e[f"k_loop_bs{bs}_err"] = float((o.float() - ref.float()).abs().max())
            except Exception:
                e[f"k_loop_bs{bs}_err"] = traceback.format_exc()[-300:]
        for bs in (512, 1024, 2048):
            if bs >= BLOCK:
                continue
            nch = triton.cdiv(N - 1, bs)
            part = torch.empty((rows * nch,), dtype=torch.float32, device=DEV)

            def two():
                k_chunksum[(rows * nch,)](y, part, N, BLOCK=bs, NCH=nch)
                k_apply[(rows * nch,)](y, o, part, N, BLOCK=bs, NCH=nch)

            try:
                e[f"k_two_bs{bs}_ms"] = bench(two)
                e[f"k_two_bs{bs}_err"] = float((o.float() - ref.float()).abs().max())
            except Exception:
                e[f"k_two_bs{bs}_err"] = traceback.format_exc()[-300:]
        res[name] = e
    OUT["bench_rows"] = res


def main():
    part_a()
    try:
        part_b()
    except Exception:
        OUT["part_b_err"] = traceback.format_exc()
    try:
        part_c()
    except Exception:
        OUT["part_c_err"] = traceback.format_exc()
    with open(os.path.join(ART, "probe.json"), "w") as f:
        json.dump(OUT, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in OUT.items() if k != "src"}, indent=1)[:8000])


main()
