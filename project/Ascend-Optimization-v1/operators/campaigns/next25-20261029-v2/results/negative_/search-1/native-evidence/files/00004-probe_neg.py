"""Bounded diagnostics: launch-overhead floor, BLOCK sweep, stride-path check."""
import json
import os
import time
import traceback

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
DEV = "npu"
res = {"torch": torch.__version__, "triton": triton.__version__}


@triton.jit
def neg_flat(X, n, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(X + offs)
        tl.store(X + offs, -x)
    else:
        m = offs < n
        x = tl.load(X + offs, mask=m)
        tl.store(X + offs, -x, mask=m)


@triton.jit
def neg_gs(X, n, n_iter, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    step = npg * BLOCK
    base = pid * BLOCK
    for _ in range(n_iter):
        offs = base + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(X + offs, mask=m)
        tl.store(X + offs, -x, mask=m)
        base += step


@triton.jit
def neg_strided(X, n, d1, d2, d3, s0, s1, s2, s3, RANK: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < n
    if RANK == 1:
        off = idx * s0
    elif RANK == 2:
        r = idx // d1
        off = r * s0 + (idx - r * d1) * s1
    elif RANK == 3:
        r0 = idx // (d1 * d2)
        rem = idx - r0 * (d1 * d2)
        r1 = rem // d2
        off = r0 * s0 + r1 * s1 + (rem - r1 * d2) * s2
    else:
        r0 = idx // (d1 * d2 * d3)
        rem0 = idx - r0 * (d1 * d2 * d3)
        r1 = rem0 // (d2 * d3)
        rem1 = rem0 - r1 * (d2 * d3)
        r2 = rem1 // d3
        off = r0 * s0 + r1 * s1 + r2 * s2 + (rem1 - r2 * d3) * s3
    x = tl.load(X + off, mask=m)
    tl.store(X + off, -x, mask=m)


def collapse(t):
    dims = [(sz, st) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    out = []
    for sz, st in reversed(dims):
        if out and out[-1][1] == st * sz:
            out[-1] = (out[-1][0] * sz, st)
        else:
            out.append((sz, st))
    out.reverse()
    return out


def run_cand(A, block=4096, mode="flat"):
    n = A.numel()
    if n == 0:
        return A
    if A.is_contiguous():
        if mode == "flat":
            grid = ((n + block - 1) // block,)
            neg_flat[grid](A, n, BLOCK=block, EVEN=(n % block == 0))
        else:
            grid = (mode,)
            n_iter = (n + block * mode - 1) // (block * mode)
            neg_gs[grid](A, n, n_iter, BLOCK=block)
        return A
    dims = collapse(A)
    rank = len(dims)
    if rank > 4:
        raise ValueError("rank>4")
    pad = dims + [(1, 0)] * (4 - rank)
    grid = ((n + block - 1) // block,)
    neg_strided[grid](A, n, pad[1][0], pad[2][0], pad[3][0],
                      pad[0][1], pad[1][1], pad[2][1], pad[3][1],
                      RANK=rank, BLOCK=block)
    return A


def bench_wall(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def bench_dev(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    s = torch.npu.Event(enable_timing=True)
    e = torch.npu.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.npu.synchronize()
    return s.elapsed_time(e) / iters


# ---------------- correctness over the frozen correctness specs -------------
CORR = [
    ("float16", [1], False, 1729), ("float16", [1], True, 1730),
    ("float16", [1025], False, 1731), ("float16", [1025], True, 1732),
    ("float16", [7, 9], False, 1733), ("float16", [7, 9], True, 1734),
    ("bfloat16", [1], False, 1735), ("bfloat16", [1], True, 1736),
    ("bfloat16", [1025], False, 1737), ("bfloat16", [1025], True, 1738),
    ("bfloat16", [7, 9], False, 1739), ("bfloat16", [7, 9], True, 1740),
    ("float32", [1], False, 1741), ("float32", [1], True, 1742),
    ("float32", [1025], False, 1743), ("float32", [1025], True, 1744),
    ("float32", [7, 9], False, 1745), ("float32", [7, 9], True, 1746),
]


def gen(shape, dt, strided, seed):
    dtype = getattr(torch, dt)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g).to(device=DEV, dtype=dtype)
    if x.numel() >= 7:
        edge = [0.0, -0.0, float("nan"), float("inf"), -float("inf"), -1.0, 1.0]
        x.flatten()[:7] = torch.tensor(edge, dtype=dtype, device=DEV)
    if strided:
        base = torch.empty((*x.shape, 2), dtype=dtype, device=DEV)
        base[..., 0] = x
        x = base[..., 0]
    return x


def dup(t, strided):
    if not strided:
        return t.clone()
    base = torch.empty((*t.shape, 2), dtype=t.dtype, device=DEV)
    base[..., 0] = t
    return base[..., 0]


corr = []
for dt, shape, strided, seed in CORR:
    try:
        a = gen(shape, dt, strided, seed)
        r = dup(a, strided)
        c = dup(a, strided)
        torch.neg_(r)
        out = run_cand(c)
        same_obj = out is c
        ok_val = bool(torch.allclose(c, r, rtol=1e-2, atol=1e-2, equal_nan=True))
        fin_c = torch.isfinite(c)
        fin_r = torch.isfinite(r)
        ok_fin = bool(torch.equal(fin_c, fin_r))
        # also check signbits of zeros preserved: reference neg(-0.)=+0.
        ok_signed = bool(torch.equal(torch.signbit(c.float()), torch.signbit(r.float())))
        corr.append({"dt": dt, "shape": shape, "strided": strided, "same_obj": same_obj,
                     "values": ok_val, "finite_mask": ok_fin, "signbit": ok_signed})
    except Exception:
        corr.append({"dt": dt, "shape": shape, "strided": strided, "error": traceback.format_exc()[-600:]})
res["correctness"] = corr

# ---------------- timing ----------------
TIM = [
    ("float16", [1024], 500), ("float16", [65536], 500), ("float16", [1048576], 200),
    ("float16", [4096, 4096], 100), ("float32", [1024], 500), ("float32", [4096, 4096], 100),
]
tim = []
for dt, shape, iters in TIM:
    x = torch.randn(shape, device=DEV, dtype=getattr(torch, dt))
    entry = {"dt": dt, "shape": shape, "n": x.numel()}
    entry["torch_neg_"] = bench_wall(lambda: torch.neg_(x), iters)
    entry["torch_neg__dev"] = bench_dev(lambda: torch.neg_(x), iters)
    for block in (1024, 4096, 8192):
        try:
            entry[f"flat_b{block}"] = bench_wall(lambda b=block: run_cand(x, b, "flat"), iters)
        except Exception:
            entry[f"flat_b{block}"] = traceback.format_exc()[-300:]
    entry["flat_b8192_dev"] = bench_dev(lambda: run_cand(x, 8192, "flat"), iters)
    for npg, block in ((20, 8192), (40, 8192), (80, 4096)):
        try:
            n_iter = (x.numel() + block * npg - 1) // (block * npg)
            entry[f"gs_p{npg}_b{block}"] = bench_wall(
                lambda p=npg, b=block: neg_gs[(p,)](x, x.numel(), (x.numel() + b * p - 1) // (b * p), BLOCK=b),
                iters)
        except Exception:
            entry[f"gs_p{npg}_b{block}"] = traceback.format_exc()[-300:]
    tim.append(entry)
res["timing"] = tim

# ---------------- launch-overhead floor ----------------
try:
    one = torch.randn([1], device=DEV, dtype=torch.float16)
    ov = {}
    ov["flat_grid1_n1"] = bench_wall(lambda: neg_flat[(1,)](one, 1, BLOCK=1024, EVEN=True), 500)
    ov["torch_neg__n1"] = bench_wall(lambda: torch.neg_(one), 500)
    big = torch.randn([1024], device=DEV, dtype=torch.float16)
    ov["flat_grid1_n1024"] = bench_wall(lambda: neg_flat[(1,)](big, 1024, BLOCK=1024, EVEN=True), 500)
    # direct compiled-kernel call (skips JIT arg binding)
    try:
        ck = neg_flat.warmup(one, 1, BLOCK=1024, EVEN=True, grid=(1,))
        ck._init_handles()
        ov["direct_ck_n1"] = bench_wall(lambda: ck[(1, 1, 1)](one, 1), 500)
    except Exception:
        ov["direct_ck_n1"] = traceback.format_exc()[-500:]
    res["overhead"] = ov
except Exception:
    res["overhead"] = traceback.format_exc()[-800:]

# ---------------- flag_gems comparison ----------------
try:
    from flag_gems.ops.neg import neg_ as fg_neg
    x = torch.randn([1024], device=DEV, dtype=torch.float16)
    res["flag_gems_n1024"] = bench_wall(lambda: fg_neg(x), 500)
    xb = torch.randn([4096, 4096], device=DEV, dtype=torch.float32)
    res["flag_gems_fp32_4096"] = bench_wall(lambda: fg_neg(xb), 100)
except Exception:
    res["flag_gems"] = traceback.format_exc()[-500:]

with open(os.path.join(ART, "neg_probe.json"), "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res, indent=1)[:12000])
