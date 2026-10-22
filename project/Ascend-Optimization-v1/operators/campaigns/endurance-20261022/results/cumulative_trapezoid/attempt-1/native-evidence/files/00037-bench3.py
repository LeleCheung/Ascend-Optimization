"""Compare chunked-scan architectures for cumulative_trapezoid on the slow cases."""
import json
import os
import time

import torch
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
res = {"cases": [], "errors": []}


# ---------- rows: scan axis is last, chunked with carry (broadcast carry) ----
@triton.jit
def k_rows_chunk(y_ptr, out_ptr, N: tl.constexpr, BR: tl.constexpr,
                 L: tl.constexpr, SCALE: tl.constexpr, SINGLE: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    k = tl.arange(0, L)
    carry = tl.zeros((BR,), dtype=tl.float32)
    n_out = N - 1
    for c in range(0, tl.cdiv(n_out, L)):
        kk = c * L + k
        m = kk < n_out
        base = r[:, None] * N
        if SINGLE:
            m2 = kk < N
            v = tl.load(y_ptr + base + kk[None, :], mask=m2[None, :], other=0.0).to(tl.float32)
            s = tl.cumsum(v, axis=1) + carry[:, None]
            y0 = tl.sum(tl.where(k[None, :] == 0, v, 0.0), axis=1)
            o = (s - 0.5 * y0[:, None] + 0.5 * v) * SCALE
            # y0 is only valid for chunk 0
            o = tl.where(c == 0, o, (s + 0.5 * v) * SCALE)
            tl.store(out_ptr + r[:, None] * n_out + kk[None, :], o, mask=m[None, :])
            carry = tl.sum(tl.where(k[None, :] == (L - 1), o / SCALE, 0.0), axis=1)
        else:
            a = tl.load(y_ptr + base + kk[None, :], mask=m[None, :], other=0.0).to(tl.float32)
            b = tl.load(y_ptr + base + (kk + 1)[None, :], mask=m[None, :], other=0.0).to(tl.float32)
            t = (a + b) * SCALE
            cs = tl.cumsum(t, axis=1) + carry[:, None]
            tl.store(out_ptr + r[:, None] * n_out + kk[None, :], cs, mask=m[None, :])
            carry = tl.sum(tl.where(k[None, :] == (L - 1), cs, 0.0), axis=1)


# ---------- cols: scan axis is first (strided), chunked with carry ----------
@triton.jit
def k_cols_chunk(y_ptr, out_ptr, N: tl.constexpr, INNER: tl.constexpr,
                 BI: tl.constexpr, L: tl.constexpr, SCALE: tl.constexpr):
    i = tl.program_id(0) * BI + tl.arange(0, BI)
    im = i < INNER
    k = tl.arange(0, L)
    carry = tl.zeros((BI,), dtype=tl.float32)
    n_out = N - 1
    for c in range(0, tl.cdiv(n_out, L)):
        kk = c * L + k
        m = (kk < n_out)[:, None] & im[None, :]
        a = tl.load(y_ptr + kk[:, None] * INNER + i[None, :], mask=m, other=0.0).to(tl.float32)
        b = tl.load(y_ptr + (kk + 1)[:, None] * INNER + i[None, :], mask=m, other=0.0).to(tl.float32)
        t = (a + b) * SCALE
        cs = tl.cumsum(t, axis=0) + carry[None, :]
        tl.store(out_ptr + kk[:, None] * INNER + i[None, :], cs, mask=m)
        carry = tl.sum(tl.where(k[:, None] == (L - 1), cs, 0.0), axis=0)


def wall(fn, iters=50, warm=10):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def dev(fn, iters=50, warm=10):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    e0 = torch.npu.Event(enable_timing=True)
    e1 = torch.npu.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.npu.synchronize()
    return e0.elapsed_time(e1) / iters


def add(name, fn, out, ref):
    try:
        fn()
        torch.npu.synchronize()
        err = (out.float() - ref.float()).abs().max().item()
        res["cases"].append({"case": name, "wall_ms": wall(fn), "dev_ms": dev(fn), "maxabs": err})
    except Exception as e:
        res["errors"].append({"case": name, "err": repr(e)[-700:]})


def main():
    torch.npu.set_device(0)

    # ===== rows, (1024, 4096)
    N = 4096
    for dt in ("float16", "float32"):
        tdt = getattr(torch, dt)
        y = torch.randn(1024, N, generator=torch.Generator().manual_seed(1730)).to(tdt).npu()
        ref = torch.cumulative_trapezoid(y, dx=1.0, dim=-1)
        res["cases"].append({"case": f"torchREF-{dt}-1024x4096", "wall_ms": wall(lambda: torch.cumulative_trapezoid(y, dx=1.0, dim=-1))})
        combos = [(1, 1024, 0), (2, 1024, 0), (4, 1024, 0), (2, 512, 0), (4, 512, 0), (8, 512, 0), (4, 1024, 1)] if dt == "float16" else [(2, 1024, 0), (4, 1024, 0), (4, 512, 0)]
        for br, L, single in combos:
            out = torch.empty(1024, N - 1, dtype=tdt, device="npu:0")
            grid = (triton.cdiv(1024, br),)
            name = f"rows-{dt}-1024x4096-br{br}-L{L}-s{single}"
            add(name, lambda br=br, L=L, single=single, out=out, y=y, grid=grid:
                k_rows_chunk[grid](y, out, N=N, BR=br, L=L, SCALE=0.5, SINGLE=single), out, ref)

    # ===== rows, (128, 1024) small case
    N = 1024
    for dt in ("float16", "float32"):
        tdt = getattr(torch, dt)
        y = torch.randn(128, N, generator=torch.Generator().manual_seed(1729)).to(tdt).npu()
        ref = torch.cumulative_trapezoid(y, dx=1.0, dim=-1)
        res["cases"].append({"case": f"torchREF-{dt}-128x1024", "wall_ms": wall(lambda: torch.cumulative_trapezoid(y, dx=1.0, dim=-1))})
        combos = [(4, 1024, 0), (8, 1024, 0), (4, 512, 0), (8, 512, 0), (16, 512, 0), (8, 256, 0)] if dt == "float16" else [(4, 1024, 0), (8, 1024, 0), (8, 512, 0)]
        for br, L, single in combos:
            out = torch.empty(128, N - 1, dtype=tdt, device="npu:0")
            grid = (triton.cdiv(128, br),)
            name = f"rows-{dt}-128x1024-br{br}-L{L}"
            add(name, lambda br=br, L=L, out=out, y=y, grid=grid:
                k_rows_chunk[grid](y, out, N=N, BR=br, L=L, SCALE=0.5, SINGLE=0), out, ref)

    # ===== cols, (1024, 128)
    N, INNER = 1024, 128
    for dt in ("float16", "float32", "bfloat16"):
        tdt = getattr(torch, dt)
        y = torch.randn(N, INNER, generator=torch.Generator().manual_seed(1731)).to(tdt).npu()
        ref = torch.cumulative_trapezoid(y, dx=1.0, dim=0)
        res["cases"].append({"case": f"torchREF-{dt}-1024x128-dim0", "wall_ms": wall(lambda: torch.cumulative_trapezoid(y, dx=1.0, dim=0))})
        combos = [(128, 128), (64, 128), (64, 256), (32, 256), (128, 64)]
        for bi, L in combos:
            out = torch.empty(N - 1, INNER, dtype=tdt, device="npu:0")
            grid = (triton.cdiv(INNER, bi),)
            name = f"cols-{dt}-1024x128-bi{bi}-L{L}"
            add(name, lambda bi=bi, L=L, out=out, y=y, grid=grid:
                k_cols_chunk[grid](y, out, N=N, INNER=INNER, BI=bi, L=L, SCALE=0.5), out, ref)

    with open(os.path.join(ART, "bench3.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1)[:24000])


if __name__ == "__main__":
    main()
