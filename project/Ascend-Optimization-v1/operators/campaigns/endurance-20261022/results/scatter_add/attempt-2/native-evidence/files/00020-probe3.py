"""Diagnostic sweep for scatter_add: maskless atomic store loop vs pipelined
owner-write kernel variants, on the two large frozen timing shapes.

Diagnostic only.
"""

import json
import os
import time

import torch
import triton
import triton.language as tl

OUTDIR = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
DEV = "npu"
torch.npu.set_device(0)
RES = {"variants": []}


def gen(seed, rows, ncols, dtype):
    g = torch.Generator().manual_seed(seed)
    inp = torch.randn(rows, ncols, generator=g).to(dtype=dtype, device=DEV)
    src = torch.randn(rows, ncols, generator=g).to(dtype=dtype, device=DEV)
    idx = torch.randint(ncols, (rows, ncols), generator=g).to(DEV)
    return inp, idx, src


def bench(fn, reps, warmup=2):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def cmp(ref, out):
    r = ref.float().reshape(-1)
    o = out.float().reshape(-1)
    d = (r - o).abs()
    lim = 0.015 + 0.015 * r.abs()
    return {"mismatched": int((d > lim).sum()), "max_abs": float(d.max())}


# maskless atomic (grid covers ncols exactly)
@triton.jit
def k_atomic_nomask(src_ptr, idx_ptr, out_ptr, ncols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    base = row * ncols
    idx = tl.load(idx_ptr + base + c).to(tl.int32)
    val = tl.load(src_ptr + base + c).to(out_ptr.dtype.element_ty)
    tl.atomic_add(out_ptr + (base + idx), val, sem="relaxed")


@triton.jit(do_not_specialize=["ncols"])
def k_atomic_mask(src_ptr, idx_ptr, out_ptr, ncols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = c < ncols
    base = row * ncols
    idx = tl.load(idx_ptr + base + c, mask=mask, other=0).to(tl.int32)
    val = tl.load(src_ptr + base + c, mask=mask, other=0).to(out_ptr.dtype.element_ty)
    tl.atomic_add(out_ptr + (base + idx), val, mask=mask, sem="relaxed")


# owner kernel, base (attempt-1 shape)
@triton.jit(do_not_specialize=["ncols"])
def k_owner(inp_ptr, idx_ptr, src_ptr, out_ptr, ncols, BLOCK: tl.constexpr, CHUNK: tl.constexpr):
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


# owner kernel, no smask in the compare (idx loaded with other=-1)
@triton.jit(do_not_specialize=["ncols"])
def k_owner_nosm(inp_ptr, idx_ptr, src_ptr, out_ptr, ncols, BLOCK: tl.constexpr, CHUNK: tl.constexpr):
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
        hit = idx[:, None] == dst[None, :]
        acc += tl.sum(tl.where(hit, val[:, None], 0.0), axis=0)
    cur = tl.load(inp_ptr + base + dst, mask=dmask, other=0).to(tl.float32)
    tl.store(out_ptr + base + dst, (cur + acc).to(out_ptr.dtype.element_ty), mask=dmask)


# owner kernel with software pipelining hint
@triton.jit(do_not_specialize=["ncols"])
def k_owner_pipe(
    inp_ptr, idx_ptr, src_ptr, out_ptr, ncols,
    BLOCK: tl.constexpr, CHUNK: tl.constexpr, STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    dst = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    dmask = dst < ncols
    base = row * ncols
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s0 in tl.range(0, ncols, CHUNK, num_stages=STAGES):
        so = s0 + tl.arange(0, CHUNK)
        smask = so < ncols
        idx = tl.load(idx_ptr + base + so, mask=smask, other=-1).to(tl.int32)
        val = tl.load(src_ptr + base + so, mask=smask, other=0).to(tl.float32)
        hit = idx[:, None] == dst[None, :]
        acc += tl.sum(tl.where(hit, val[:, None], 0.0), axis=0)
    cur = tl.load(inp_ptr + base + dst, mask=dmask, other=0).to(tl.float32)
    tl.store(out_ptr + base + dst, (cur + acc).to(out_ptr.dtype.element_ty), mask=dmask)


# owner kernel, 2-way manually unrolled chunks (two independent load streams)
@triton.jit(do_not_specialize=["ncols"])
def k_owner_u2(inp_ptr, idx_ptr, src_ptr, out_ptr, ncols, BLOCK: tl.constexpr, CHUNK: tl.constexpr):
    row = tl.program_id(0)
    dst = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    dmask = dst < ncols
    base = row * ncols
    acc0 = tl.zeros([BLOCK], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK], dtype=tl.float32)
    step: tl.constexpr = 2 * CHUNK
    for s0 in range(0, ncols, step):
        so0 = s0 + tl.arange(0, CHUNK)
        so1 = s0 + CHUNK + tl.arange(0, CHUNK)
        i0 = tl.load(idx_ptr + base + so0, mask=so0 < ncols, other=-1).to(tl.int32)
        i1 = tl.load(idx_ptr + base + so1, mask=so1 < ncols, other=-1).to(tl.int32)
        v0 = tl.load(src_ptr + base + so0, mask=so0 < ncols, other=0).to(tl.float32)
        v1 = tl.load(src_ptr + base + so1, mask=so1 < ncols, other=0).to(tl.float32)
        acc0 += tl.sum(tl.where(i0[:, None] == dst[None, :], v0[:, None], 0.0), axis=0)
        acc1 += tl.sum(tl.where(i1[:, None] == dst[None, :], v1[:, None], 0.0), axis=0)
    acc = acc0 + acc1
    cur = tl.load(inp_ptr + base + dst, mask=dmask, other=0).to(tl.float32)
    tl.store(out_ptr + base + dst, (cur + acc).to(out_ptr.dtype.element_ty), mask=dmask)


def run_variant(name, fn, ref, out):
    try:
        fn()
        torch.npu.synchronize()
        chk = cmp(ref, out)
        ms = bench(fn, REPS, warmup=1)
        RES["variants"].append({"shape": SHAPE, "name": name, "ms": ms, "check": chk})
        print("VAR", SHAPE, name, round(ms, 5), chk, flush=True)
    except Exception as exc:  # noqa: BLE001
        RES["variants"].append({"shape": SHAPE, "name": name, "error": repr(exc)[:300]})
        print("VAR-FAIL", SHAPE, name, repr(exc)[:200], flush=True)


for SHAPE, REPS, seed in (((128, 128), 20, 1729), ((1024, 1024), 5, 1730), ((2048, 2048), 2, 1731)):
    rows, ncols = SHAPE
    inp, index, src = gen(seed, rows, ncols, torch.float16)
    ref = torch.scatter_add(inp, 0, index, src) if False else torch.scatter_add(
        inp, -1, index, src
    )

    out = torch.empty_like(inp)
    run_variant(
        "atomic_mask_1024",
        lambda: k_atomic_mask[(rows, triton.cdiv(ncols, 1024))](src, index, out, ncols, BLOCK=1024),
        ref, out,
    )
    if ncols % 1024 == 0:
        out = torch.empty_like(inp)
        run_variant(
            "atomic_nomask_1024",
            lambda: k_atomic_nomask[(rows, ncols // 1024)](src, index, out, ncols, BLOCK=1024),
            ref, out,
        )
    if ncols % 2048 == 0:
        out = torch.empty_like(inp)
        run_variant(
            "atomic_nomask_2048",
            lambda: k_atomic_nomask[(rows, ncols // 2048)](src, index, out, ncols, BLOCK=2048),
            ref, out,
        )

    out = torch.empty_like(inp)
    run_variant(
        "owner_256_32",
        lambda: k_owner[(rows, triton.cdiv(ncols, 256))](inp, index, src, out, ncols, BLOCK=256, CHUNK=32),
        ref, out,
    )
    out = torch.empty_like(inp)
    run_variant(
        "owner_nosm_256_32",
        lambda: k_owner_nosm[(rows, triton.cdiv(ncols, 256))](inp, index, src, out, ncols, BLOCK=256, CHUNK=32),
        ref, out,
    )
    out = torch.empty_like(inp)
    run_variant(
        "owner_pipe_256_32_s3",
        lambda: k_owner_pipe[(rows, triton.cdiv(ncols, 256))](
            inp, index, src, out, ncols, BLOCK=256, CHUNK=32, STAGES=3
        ),
        ref, out,
    )
    out = torch.empty_like(inp)
    run_variant(
        "owner_u2_256_32",
        lambda: k_owner_u2[(rows, triton.cdiv(ncols, 256))](inp, index, src, out, ncols, BLOCK=256, CHUNK=32),
        ref, out,
    )
    out = torch.empty_like(inp)
    run_variant(
        "owner_nosm_128_64",
        lambda: k_owner_nosm[(rows, triton.cdiv(ncols, 128))](inp, index, src, out, ncols, BLOCK=128, CHUNK=64),
        ref, out,
    )
    out = torch.empty_like(inp)
    run_variant(
        "owner_nosm_64_128",
        lambda: k_owner_nosm[(rows, triton.cdiv(ncols, 64))](inp, index, src, out, ncols, BLOCK=64, CHUNK=128),
        ref, out,
    )
    # sanity re-measure after any compile failure
    out = torch.empty_like(inp)
    run_variant(
        "owner_256_32_remeasure",
        lambda: k_owner[(rows, triton.cdiv(ncols, 256))](inp, index, src, out, ncols, BLOCK=256, CHUNK=32),
        ref, out,
    )

path = os.path.join(OUTDIR, "probe3.json")
with open(path, "w", encoding="utf-8") as fh:
    json.dump(RES, fh, indent=1)
print("WROTE", path, flush=True)
