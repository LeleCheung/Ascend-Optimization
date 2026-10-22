"""Bounded target-side microbenchmark for cumulative_trapezoid kernel shapes.

Measures full-call walltime (host + device), matching the evaluator's timing
mode, for candidate tilings of the scan. Also checks numerics against
torch.cumulative_trapezoid (reference use only; not an implementation).
"""
import json
import os
import time

import torch
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")


# ---------------- last-dim (inner == 1): batched rows -------------------
@triton.jit
def k_rows(y_ptr, out_ptr, N, BR: tl.constexpr, BLOCK: tl.constexpr, SCALE):
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    k = tl.arange(0, BLOCK)
    mask = k < (N - 1)
    base = r[:, None] * N
    a = tl.load(y_ptr + base + k[None, :], mask=mask[None, :], other=0.0).to(tl.float32)
    b = tl.load(y_ptr + base + 1 + k[None, :], mask=mask[None, :], other=0.0).to(tl.float32)
    t = (a + b) * SCALE
    cs = tl.cumsum(t, axis=1)
    tl.store(out_ptr + r[:, None] * (N - 1) + k[None, :], cs, mask=mask[None, :])


# single-load variant: cumsum of y then endpoint correction
@triton.jit
def k_rows_single(y_ptr, out_ptr, N, BR: tl.constexpr, BLOCK: tl.constexpr, SCALE):
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    k = tl.arange(0, BLOCK)
    mask = k < N
    base = r[:, None] * N
    v = tl.load(y_ptr + base + k[None, :], mask=mask[None, :], other=0.0).to(tl.float32)
    s = tl.cumsum(v, axis=1)
    y0 = tl.sum(tl.where(k[None, :] == 0, v, 0.0), axis=1)
    outv = (s - 0.5 * y0[:, None] + 0.5 * v) * SCALE
    m2 = k < (N - 1)
    tl.store(out_ptr + r[:, None] * (N - 1) + k[None, :], outv, mask=m2[None, :])


# ---------------- non-last dim: batch over inner, scan rows -------------
@triton.jit
def k_cols(y_ptr, out_ptr, N, INNER, BI: tl.constexpr, L: tl.constexpr, SCALE):
    i = tl.program_id(0) * BI + tl.arange(0, BI)
    im = i < INNER
    k = tl.arange(0, L)
    carry = tl.zeros((L, BI), dtype=tl.float32)
    nchunk = tl.cdiv(N - 1, L)
    for c in range(0, nchunk):
        kk = c * L + k
        m = (kk < (N - 1)) & im[None, :]
        a = tl.load(y_ptr + kk[:, None] * INNER + i[None, :], mask=m, other=0.0).to(tl.float32)
        b = tl.load(y_ptr + (kk + 1)[:, None] * INNER + i[None, :], mask=m, other=0.0).to(tl.float32)
        t = (a + b) * SCALE
        cs = tl.cumsum(t, axis=0) + carry
        tl.store(out_ptr + kk[:, None] * INNER + i[None, :], cs, mask=m)
        last = tl.sum(tl.where(k[:, None] == (L - 1), cs, 0.0), axis=0)
        carry = last[None, :] + tl.zeros((L, BI), dtype=tl.float32)


@triton.jit
def k_cols_t(y_ptr, out_ptr, N, INNER, BI: tl.constexpr, L: tl.constexpr, SCALE):
    i = tl.program_id(0) * BI + tl.arange(0, BI)
    im = i < INNER
    k = tl.arange(0, L)
    carry = tl.zeros((BI, L), dtype=tl.float32)
    nchunk = tl.cdiv(N - 1, L)
    for c in range(0, nchunk):
        kk = c * L + k
        m = (kk < (N - 1)) & im[:, None]
        a = tl.load(y_ptr + kk[None, :] * INNER + i[:, None], mask=m, other=0.0).to(tl.float32)
        b = tl.load(y_ptr + (kk + 1)[None, :] * INNER + i[:, None], mask=m, other=0.0).to(tl.float32)
        t = (a + b) * SCALE
        cs = tl.cumsum(t, axis=1) + carry
        tl.store(out_ptr + kk[None, :] * INNER + i[:, None], cs, mask=m)
        last = tl.sum(tl.where(k[None, :] == (L - 1), cs, 0.0), axis=1)
        carry = last[:, None] + tl.zeros((BI, L), dtype=tl.float32)


@triton.jit
def k_nop(x_ptr):
    tl.store(x_ptr + tl.program_id(0), 0.0)


def wall(fn, iters=30, warm=5):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def ref(y, dx, dim):
    return torch.cumulative_trapezoid(y, dx=dx, dim=dim)


def main():
    torch.npu.set_device(0)
    res = {"cases": [], "launch_floor_ms": None, "errors": []}

    z = torch.zeros(4096, dtype=torch.float32, device="npu:0")
    try:
        res["launch_floor_ms"] = wall(lambda: k_nop[(64,)](z))
    except Exception as e:
        res["errors"].append({"what": "nop", "err": repr(e)[:300]})

    plan = {
        ("float16", (128, 1024)): (("two", "one"), (4, 8, 16, 32)),
        ("float16", (1024, 4096)): (("two", "one"), (1, 2, 4)),
        ("float32", (128, 1024)): (("two",), (8, 16, 32)),
        ("float32", (1024, 4096)): (("two",), (1, 2, 4)),
    }
    kerns = {"two": k_rows, "one": k_rows_single}
    for (dt, shape), (tags, brs) in plan.items():
        tdt = getattr(torch, dt)
        y = torch.randn(shape, generator=torch.Generator().manual_seed(1729)).to(tdt).npu()
        r = ref(y, 1, -1)
        N = shape[1]
        BLOCK = triton.next_power_of_2(N)
        for br in brs:
            for tag in tags:
                kern = kerns[tag]
                out = torch.empty(shape[0], N - 1, dtype=tdt, device="npu:0")
                grid = (triton.cdiv(shape[0], br),)
                name = f"rows-{dt}-{shape[0]}x{N}-br{br}-{tag}"
                try:
                    def fn(kern=kern, grid=grid, br=br, out=out, y=y, N=N, BLOCK=BLOCK):
                        kern[grid](y, out, N, BR=br, BLOCK=BLOCK, SCALE=0.5)
                    fn()
                    torch.npu.synchronize()
                    err = (out.float() - r.float()).abs().max().item()
                    res["cases"].append({"case": name, "ms": wall(fn), "maxabs": err})
                except Exception as e:
                    res["errors"].append({"case": name, "err": repr(e)[:300]})
        res["cases"].append({"case": f"torchREF-{dt}-{shape[0]}x{N}", "ms": wall(lambda y=y: ref(y, 1, -1))})

    for dt in ("float16", "float32"):
        tdt = getattr(torch, dt)
        shape = (1024, 128)
        y = torch.randn(shape, generator=torch.Generator().manual_seed(1731)).to(tdt).npu()
        r = ref(y, 1, 0)
        for bi, L, kern, tag in ((128, 128, k_cols, "ax0"), (128, 256, k_cols, "ax0"),
                                 (64, 128, k_cols, "ax0"), (128, 256, k_cols_t, "t-ax1"),
                                 (64, 256, k_cols_t, "t-ax1")):
            out = torch.empty(shape[0] - 1, shape[1], dtype=tdt, device="npu:0")
            grid = (triton.cdiv(shape[1], bi),)
            name = f"cols-{dt}-1024x128-bi{bi}-L{L}-{tag}"
            try:
                def fn(kern=kern, grid=grid, out=out, y=y, bi=bi, L=L):
                    kern[grid](y, out, shape[0], shape[1], BI=bi, L=L, SCALE=0.5)
                fn()
                torch.npu.synchronize()
                err = (out.float() - r.float()).abs().max().item()
                res["cases"].append({"case": name, "ms": wall(fn), "maxabs": err})
            except Exception as e:
                res["errors"].append({"case": name, "err": repr(e)[:300]})
        res["cases"].append({"case": f"torchREF-{dt}-1024x128-dim0", "ms": wall(lambda y=y: ref(y, 1, 0))})

    with open(os.path.join(ART, "bench.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1)[:20000])


if __name__ == "__main__":
    main()
