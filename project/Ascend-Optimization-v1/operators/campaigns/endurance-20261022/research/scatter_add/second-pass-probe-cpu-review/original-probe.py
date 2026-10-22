"""Bounded NPU probe for scatter_add.

Part 1 (numerics): identify the accumulation semantics of the device reference
(torch.scatter_add on NPU) by comparing it against explicitly emulated models
(fp64 exact, fp32 sequential, low-precision sequential) on the frozen contract
inputs.

Part 2 (performance): micro-benchmark candidate kernel architectures on the
frozen timing shapes (walltime, sync-per-rep).

Diagnostic only: not a formal evaluation, not a candidate implementation.
"""

import json
import os
import time

import numpy as np
import torch
import triton
import triton.language as tl

OUTDIR = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
DEV = "npu"
torch.npu.set_device(0)

RESULT = {"device": torch.npu.get_device_name(0), "numerics": [], "perf": [], "notes": []}


# --------------------------------------------------------------------------- #
# inputs (identical to the frozen contract generator)
# --------------------------------------------------------------------------- #
def gen(seed, shape, dim, dtype):
    g = torch.Generator().manual_seed(seed)

    def random(sh):
        return torch.randn(sh, generator=g).to(dtype=dtype, device=DEV)

    x = random(shape)
    s = random(shape)
    idx = torch.randint(shape[dim % len(shape)], shape, generator=g).to(DEV)
    return x, dim, idx, s


# --------------------------------------------------------------------------- #
# rounding helpers (CPU, exact-then-round)
# --------------------------------------------------------------------------- #
def _bf16_rne(f32):
    f32 = np.ascontiguousarray(f32, dtype=np.float32)
    u = f32.view(np.uint32)
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    r = (u + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
    return r.view(np.float32)


def rnd_scalar(x, dtype):
    x32 = np.float32(x)
    if dtype == torch.bfloat16:
        return float(_bf16_rne(np.array([x32], dtype=np.float32))[0])
    if dtype == torch.float16:
        return float(np.float16(x32))
    return float(x32)


def rnd_tensor(t32, dtype):
    if dtype == torch.bfloat16:
        return torch.from_numpy(_bf16_rne(t32.numpy()))
    if dtype == torch.float16:
        return t32.to(torch.float16).to(torch.float32)
    return t32.to(torch.float32)


def check(ref, cand, atol, rtol):
    d = (ref - cand).abs()
    lim = atol + rtol * ref.abs()
    bad = d > lim
    return {
        "mismatched": int(bad.sum()),
        "total": int(ref.numel()),
        "max_abs": float(d.max()) if ref.numel() else 0.0,
    }


# --------------------------------------------------------------------------- #
# numerics models
# --------------------------------------------------------------------------- #
def model_small(inp, index, src, shape, dim, dtype):
    """All models for a small tensor via explicit per-element folding."""
    ndim = len(shape)
    st = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        st[i] = st[i + 1] * shape[i]
    total = int(np.prod(shape))
    coords = np.unravel_index(np.arange(total), shape)
    dst = np.zeros(total, dtype=np.int64)
    for d in range(ndim):
        src_coord = index.reshape(-1) if d == dim else coords[d]
        dst += src_coord * st[d]
    srcf = src.reshape(-1).astype(np.float64)
    inpf = inp.reshape(-1).astype(np.float64)
    order = np.argsort(dst, kind="stable")

    models = {}
    for name in ("fp64", "fp32", "lowp"):
        acc = inpf.copy()
        for t in order:
            j = dst[t]
            v = srcf[t]
            if name == "fp64":
                acc[j] = acc[j] + v
            elif name == "fp32":
                acc[j] = float(np.float32(acc[j] + v))
            else:
                acc[j] = rnd_scalar(acc[j] + v, dtype)
        models[name] = acc.astype(np.float32)
    return models, dst


def model_2d_lastdim(inp32, idx, src32, dtype, mode):
    """Column-ordered models for a contiguous (rows, cols) last-dim scatter."""
    rows, ncols = inp32.shape
    if mode == "fp64":
        acc = inp32.double()
    else:
        acc = inp32.clone()
    for c in range(ncols):
        col = idx[:, c : c + 1]
        v = src32[:, c : c + 1]
        if mode == "fp64":
            prev = acc.gather(1, col)
            acc.scatter_(1, col, prev + v.double())
        elif mode == "fp32":
            prev = acc.gather(1, col)
            acc.scatter_(1, col, prev + v)
        elif mode == "lowp":
            prev = acc.gather(1, col)
            acc.scatter_(1, col, rnd_tensor(prev + v, dtype))
        else:
            raise ValueError(mode)
    return acc.float()


def numerics_probe():
    small_cases = []
    for dt, name in ((torch.float16, "fp16"), (torch.bfloat16, "bf16"), (torch.float32, "fp32")):
        for i, dim in enumerate((0, 1, 2)):
            small_cases.append(
                dict(tag=f"{name}-3d-dim{dim}", seed=1729 + 2 * i, shape=[5, 7, 9], dim=dim, dtype=dt)
            )
    big_cases = [
        dict(tag="bf16-1024", seed=1733, shape=[1024, 1024], dim=-1, dtype=torch.bfloat16),
        dict(tag="fp16-1024", seed=1730, shape=[1024, 1024], dim=-1, dtype=torch.float16),
        dict(tag="bf16-128", seed=1732, shape=[128, 128], dim=-1, dtype=torch.bfloat16),
    ]
    atol_rtol = {torch.float16: (0.015, 0.015), torch.bfloat16: (0.015, 0.015), torch.float32: (1e-4, 1e-4)}

    for case in small_cases + big_cases:
        inp, dim, index, src = gen(case["seed"], case["shape"], case["dim"], case["dtype"])
        ref = torch.scatter_add(inp, dim, index, src)
        ref64 = ref.detach().to("cpu").to(torch.float64)
        ret = {"tag": case["tag"], "shape": case["shape"], "dim": case["dim"], "dtype": str(case["dtype"])}
        atol, rtol = atol_rtol[case["dtype"]]
        dimn = dim % len(case["shape"])
        if len(case["shape"]) == 3 or int(np.prod(case["shape"])) <= 8192:
            models, _ = model_small(
                inp.detach().to("cpu").to(torch.float32).numpy(),
                index.detach().to("cpu").numpy(),
                src.detach().to("cpu").to(torch.float32).numpy(),
                case["shape"],
                dimn,
                case["dtype"],
            )
            flat_ref = ref64.reshape(-1)
            for name, m in models.items():
                cand = torch.from_numpy(m).to(torch.float64)
                ret[name] = check(flat_ref, cand, atol, rtol)
            # detailed dump of the reference fold for the worst element
            if not ret.get("fp32", {}).get("mismatched", 0) == 0 or not ret.get("lowp", {}).get("mismatched", 0) == 0:
                ret["ref_head"] = [float(x) for x in flat_ref.reshape(-1)[:8]]
        else:
            inp32 = inp.detach().to("cpu").float()
            idx_cpu = index.detach().to("cpu")
            src32 = src.detach().to("cpu").float()
            for mode in ("fp64", "fp32", "lowp"):
                m = model_2d_lastdim(inp32, idx_cpu, src32, case["dtype"], mode)
                ret[mode] = check(ref64.reshape(-1), m.reshape(-1).to(torch.float64), atol, rtol)
        RESULT["numerics"].append(ret)
        print("NUM", json.dumps(ret), flush=True)


# --------------------------------------------------------------------------- #
# kernels (diagnostic variants)
# --------------------------------------------------------------------------- #
@triton.jit
def k_copy(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(dst_ptr + off, tl.load(src_ptr + off, mask=mask), mask=mask)


@triton.jit
def k_atomic(src_ptr, idx_ptr, out_ptr, ncols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = c < ncols
    base = row * ncols
    idx = tl.load(idx_ptr + base + c, mask=mask, other=0).to(tl.int32)
    val = tl.load(src_ptr + base + c, mask=mask, other=0)
    tl.atomic_add(out_ptr + (base + idx), val, mask=mask, sem="relaxed")


@triton.jit
def k_dense(inp_ptr, idx_ptr, src_ptr, out_ptr, ncols, BLOCK: tl.constexpr, CHUNK: tl.constexpr):
    row = tl.program_id(0)
    dst = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    dmask = dst < ncols
    base = row * ncols
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s0 in range(0, ncols, CHUNK):
        so = s0 + tl.arange(0, CHUNK)
        smask = so < ncols
        idx = tl.load(idx_ptr + base + so, mask=smask, other=-1).to(tl.int32)
        val = tl.load(src_ptr + base + so, mask=smask, other=0).to(tl.float32)
        hit = (idx[:, None] == dst[None, :]) & smask[:, None]
        acc += tl.sum(tl.where(hit, val[:, None], 0.0), axis=0)
    cur = tl.load(inp_ptr + base + dst, mask=dmask, other=0).to(tl.float32)
    tl.store(out_ptr + base + dst, (cur + acc).to(out_ptr.dtype.element_ty), mask=dmask)


@triton.jit
def k_dense_t(inp_ptr, idx_ptr, src_ptr, out_ptr, ncols, BLOCK: tl.constexpr, CHUNK: tl.constexpr):
    row = tl.program_id(0)
    dst = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    dmask = dst < ncols
    base = row * ncols
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s0 in range(0, ncols, CHUNK):
        so = s0 + tl.arange(0, CHUNK)
        smask = so < ncols
        idx = tl.load(idx_ptr + base + so, mask=smask, other=-1).to(tl.int32)
        val = tl.load(src_ptr + base + so, mask=smask, other=0).to(tl.float32)
        hit = (dst[:, None] == idx[None, :]) & smask[None, :]
        acc += tl.sum(tl.where(hit, val[None, :], 0.0), axis=1)
    cur = tl.load(inp_ptr + base + dst, mask=dmask, other=0).to(tl.float32)
    tl.store(out_ptr + base + dst, (cur + acc).to(out_ptr.dtype.element_ty), mask=dmask)


@triton.jit
def k_dense_mul(inp_ptr, idx_ptr, src_ptr, out_ptr, ncols, BLOCK: tl.constexpr, CHUNK: tl.constexpr):
    row = tl.program_id(0)
    dst = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    dmask = dst < ncols
    base = row * ncols
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s0 in range(0, ncols, CHUNK):
        so = s0 + tl.arange(0, CHUNK)
        smask = so < ncols
        idx = tl.load(idx_ptr + base + so, mask=smask, other=-1).to(tl.int32)
        val = tl.load(src_ptr + base + so, mask=smask, other=0).to(tl.float32)
        hit = ((idx[:, None] == dst[None, :]) & smask[:, None]).to(tl.float32)
        acc += tl.sum(val[:, None] * hit, axis=0)
    cur = tl.load(inp_ptr + base + dst, mask=dmask, other=0).to(tl.float32)
    tl.store(out_ptr + base + dst, (cur + acc).to(out_ptr.dtype.element_ty), mask=dmask)


# --------------------------------------------------------------------------- #
# benchmark harness
# --------------------------------------------------------------------------- #
def bench(fn, reps, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def atomic_path(inp, index, src, block):
    rows, ncols = inp.shape
    out = torch.empty_like(inp)
    n = inp.numel()
    k_copy[(triton.cdiv(n, 8192),)](inp, out, n, BLOCK=8192)
    k_atomic[(rows, triton.cdiv(ncols, block))](src, index, out, ncols, BLOCK=block)
    return out


def atomic_path_f32(inp, index, src, block):
    rows, ncols = inp.shape
    buf = torch.empty(inp.shape, dtype=torch.float32, device=inp.device)
    out = torch.empty_like(inp)
    n = inp.numel()
    k_copy[(triton.cdiv(n, 8192),)](inp, buf, n, BLOCK=8192)
    k_atomic[(rows, triton.cdiv(ncols, block))](src, index, buf, ncols, BLOCK=block)
    k_copy[(triton.cdiv(n, 8192),)](buf, out, n, BLOCK=8192)
    return out


def dense_path(kern, inp, index, src, block, chunk):
    rows, ncols = inp.shape
    out = torch.empty_like(inp)
    kern[(rows, triton.cdiv(ncols, block))](inp, index, src, out, ncols, BLOCK=block, CHUNK=chunk)
    return out


def perf_probe():
    shapes = [
        (128, 128, torch.float16, 1729, 40),
        (1024, 1024, torch.float16, 1730, 8),
        (2048, 2048, torch.float16, 1731, 3),
    ]
    for rows, ncols, dt, seed, reps in shapes:
        inp, dim, index, src = gen(seed, [rows, ncols], -1, dt)
        ref = torch.scatter_add(inp, dim, index, src)
        variants = {
            "torch_ref": lambda: torch.scatter_add(inp, dim, index, src),
            "atomic_b256": lambda: atomic_path(inp, index, src, 256),
            "atomic_b1024": lambda: atomic_path(inp, index, src, 1024),
            "atomic_f32_b1024": lambda: atomic_path_f32(inp, index, src, 1024),
            "dense_128_32": lambda: dense_path(k_dense, inp, index, src, 128, 32),
            "dense_64_64": lambda: dense_path(k_dense, inp, index, src, 64, 64),
            "dense_256_32": lambda: dense_path(k_dense, inp, index, src, 256, 32),
            "denseT_128_32": lambda: dense_path(k_dense_t, inp, index, src, 128, 32),
            "denseM_128_32": lambda: dense_path(k_dense_mul, inp, index, src, 128, 32),
        }
        for name, fn in variants.items():
            try:
                out = fn()
                torch.npu.synchronize()
                err = None
                if out.shape == ref.shape:
                    d = (out.float() - ref.float()).abs()
                    lim = 0.015 + 0.015 * ref.float().abs()
                    err = {"mismatched": int((d > lim).sum()), "max_abs": float(d.max())}
                ms = bench(fn, reps)
                RESULT["perf"].append(
                    {"shape": [rows, ncols], "dtype": str(dt), "variant": name, "ms": ms, "err": err}
                )
                print("PERF", rows, ncols, name, round(ms, 5), err, flush=True)
            except Exception as exc:  # noqa: BLE001
                RESULT["perf"].append(
                    {"shape": [rows, ncols], "dtype": str(dt), "variant": name, "error": repr(exc)[:400]}
                )
                print("PERF-FAIL", rows, ncols, name, repr(exc)[:300], flush=True)

    # host launch overhead reference: one tiny kernel launch
    tiny = torch.zeros(64, dtype=torch.float16, device=DEV)
    tiny2 = torch.zeros(64, dtype=torch.float16, device=DEV)
    ms = bench(lambda: k_copy[(1,)](tiny, tiny2, 64, BLOCK=64), 200, warmup=5)
    RESULT["launch_overhead_us"] = ms * 1e3
    print("LAUNCH_OVERHEAD_US", ms * 1e3, flush=True)


def main():
    try:
        numerics_probe()
    except Exception as exc:  # noqa: BLE001
        RESULT["numerics_error"] = repr(exc)[:800]
        print("NUMERICS_FAIL", repr(exc)[:600], flush=True)
    try:
        perf_probe()
    except Exception as exc:  # noqa: BLE001
        RESULT["perf_error"] = repr(exc)[:800]
        print("PERF_FAIL", repr(exc)[:600], flush=True)
    path = os.path.join(OUTDIR, "probe.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(RESULT, fh, indent=1)
    print("WROTE", path, flush=True)


main()
