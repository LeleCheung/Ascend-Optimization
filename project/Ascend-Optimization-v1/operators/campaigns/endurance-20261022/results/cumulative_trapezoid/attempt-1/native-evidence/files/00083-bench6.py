"""Compare mask strategies for the dot-based prefix kernel (tail handling)."""
import json
import os
import time

import torch
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
res = {"cases": [], "errors": []}


@triton.jit
def k_full(y_ptr, out_ptr, N: tl.constexpr, NOUT: tl.constexpr, ROWS: tl.constexpr,
           BR: tl.constexpr, L: tl.constexpr, SCALE: tl.constexpr):
    """No mask at all: only valid when NOUT % L == 0."""
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    c = tl.arange(0, L)
    seg = (c[:, None] >= c[None, :]).to(y_ptr.dtype.element_ty)
    carry = tl.zeros((BR,), dtype=tl.float32)
    ib = r[:, None] * N
    ob = r[:, None] * NOUT
    for cc in range(0, NOUT // L):
        kk = cc * L + c
        a = tl.load(y_ptr + ib + kk[None, :])
        b = tl.load(y_ptr + ib + 1 + kk[None, :])
        local = tl.dot(a + b, seg) * SCALE
        tl.store(out_ptr + ob + kk[None, :], local + carry[:, None])
        carry += tl.sum(tl.where(c[None, :] == (L - 1), local, 0.0), axis=1)


@triton.jit
def k_loadmask(y_ptr, out_ptr, N: tl.constexpr, NOUT: tl.constexpr, ROWS: tl.constexpr,
               BR: tl.constexpr, L: tl.constexpr, SCALE: tl.constexpr):
    """1D-expanded mask on both load and store."""
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    c = tl.arange(0, L)
    seg = (c[:, None] >= c[None, :]).to(y_ptr.dtype.element_ty)
    carry = tl.zeros((BR,), dtype=tl.float32)
    ib = r[:, None] * N
    ob = r[:, None] * NOUT
    for cc in range(0, (NOUT + L - 1) // L):
        kk = cc * L + c
        m = (kk < NOUT)[None, :]
        a = tl.load(y_ptr + ib + kk[None, :], mask=m, other=0.0)
        b = tl.load(y_ptr + ib + 1 + kk[None, :], mask=m, other=0.0)
        local = tl.dot(a + b, seg) * SCALE
        tl.store(out_ptr + ob + kk[None, :], local + carry[:, None], mask=m)
        carry += tl.sum(tl.where(c[None, :] == (L - 1), local, 0.0), axis=1)


@triton.jit
def k_clampmask(y_ptr, out_ptr, N: tl.constexpr, NOUT: tl.constexpr, ROWS: tl.constexpr,
                BR: tl.constexpr, L: tl.constexpr, SCALE: tl.constexpr):
    """Clamped load indices (always in bounds), mask only on the store."""
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    c = tl.arange(0, L)
    seg = (c[:, None] >= c[None, :]).to(y_ptr.dtype.element_ty)
    carry = tl.zeros((BR,), dtype=tl.float32)
    ib = r[:, None] * N
    ob = r[:, None] * NOUT
    for cc in range(0, (NOUT + L - 1) // L):
        kk = cc * L + c
        kc = tl.minimum(kk, NOUT - 1)
        a = tl.load(y_ptr + ib + kc[None, :])
        b = tl.load(y_ptr + ib + 1 + kc[None, :])
        local = tl.dot(a + b, seg) * SCALE
        tl.store(out_ptr + ob + kk[None, :], local + carry[:, None], mask=(kk < NOUT)[None, :])
        carry += tl.sum(tl.where(c[None, :] == (L - 1), local, 0.0), axis=1)


def wall(fn, iters=30, warm=8):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def dev(fn, iters=30, warm=8):
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


def add(name, fn, out, ref, atol):
    try:
        fn()
        torch.npu.synchronize()
        d = (out.float() - ref.float()).abs()
        bad = (d > atol + atol * ref.float().abs()).sum().item()
        res["cases"].append({"case": name, "wall_ms": wall(fn), "dev_ms": dev(fn),
                             "maxabs": d.max().item(), "bad": bad, "n": ref.numel()})
    except Exception as e:
        res["errors"].append({"case": name, "err": repr(e)[-260:]})


def main():
    torch.npu.set_device(0)
    kerns = {"full": k_full, "loadmask": k_loadmask, "clampmask": k_clampmask}
    for R, N, seed in ((1024, 4096, 1730), (128, 1024, 1729)):
        for dt in ("float16", "float32"):
            tdt = getattr(torch, dt)
            y = torch.randn(R, N, generator=torch.Generator().manual_seed(seed)).to(tdt).npu()
            ref = torch.cumulative_trapezoid(y, dx=1.0, dim=-1)
            atol = 0.015 if dt != "float32" else 1e-4
            res["cases"].append({"case": f"torchREF-{dt}-{R}x{N}",
                                 "wall_ms": wall(lambda: torch.cumulative_trapezoid(y, dx=1.0, dim=-1))})
            # padded variant: NOUT2 multiple of L, so "full" is also valid
            for kind, kern in kerns.items():
                for BR, L in ((16, 64), (16, 128), (32, 64), (32, 128)):
                    nout = N - 1
                    out = torch.empty(R, nout, dtype=tdt, device="npu:0")
                    g = (triton.cdiv(R, BR),)
                    name = f"{kind}-{dt}-{R}x{N}-br{BR}-L{L}"
                    add(name, lambda kern=kern, g=g, BR=BR, L=L, out=out, y=y, R=R, N=N, nout=nout:
                        kern[g](y, out, N=N, NOUT=nout, ROWS=R, BR=BR, L=L, SCALE=0.5), out, ref, atol)
                # exact-multiple correctness/timing probe for the mask-free kernel
                L = 64
                nout = (N - 1) // L * L
                out = torch.empty(R, nout, dtype=tdt, device="npu:0")
                ref2 = ref[:, :nout]
                g = (triton.cdiv(R, 32),)
                add(f"full-exact-{dt}-{R}x{N}-L{L}",
                    lambda g=g, out=out, y=y, R=R, N=N, nout=nout, L=L:
                        k_full[g](y, out, N=N, NOUT=nout, ROWS=R, BR=32, L=L, SCALE=0.5),
                    out, ref2, atol)
    with open(os.path.join(ART, "bench6.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1)[:22000])


if __name__ == "__main__":
    main()
