"""Bounded probe #2: compare the atomic scatter against an all-affine dense
comparison kernel (no data-dependent addresses anywhere).

For each row, out[j] = inp[j] + sum over source columns c with index[c] == j of
src[c].  The dense kernel realises the gather side as a broadcast compare over
a [CHUNK, BLOCK] tile, so every load/store address is affine and DMA friendly.
Also measures the pure load+affine-store cost of the same data volume to bound
the achievable time.
"""

import importlib.util
import json
import os
import time
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl

DEV = "npu:0"


@triton.jit(do_not_specialize=["ncols"])
def _atomic_last(src_ptr, idx_ptr, out_ptr, ncols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < ncols
    base = row * ncols
    off = base + col
    idx = tl.load(idx_ptr + off, mask=mask, other=0).to(tl.int32)
    val = tl.load(src_ptr + off, mask=mask, other=0)
    tl.atomic_add(out_ptr + (base + idx), val, mask=mask, sem="relaxed")


@triton.jit(do_not_specialize=["ncols"])
def _load_only(inp_ptr, idx_ptr, src_ptr, out_ptr, ncols, BLOCK: tl.constexpr):
    """Same data volume, fully affine addresses: the lower bound."""
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < ncols
    off = row * ncols + col
    idx = tl.load(idx_ptr + off, mask=mask, other=0).to(tl.int32)
    val = tl.load(src_ptr + off, mask=mask, other=0)
    base_v = tl.load(inp_ptr + off, mask=mask, other=0)
    acc = tl.where(idx >= 0, val, base_v)
    tl.store(out_ptr + off, acc, mask=mask)


@triton.jit(do_not_specialize=["ncols"])
def _dense_last(inp_ptr, idx_ptr, src_ptr, out_ptr, ncols, BLOCK: tl.constexpr, CHUNK: tl.constexpr):
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


def bench(fn, iters):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    wall = (time.perf_counter() - t0) / iters
    end.record()
    torch.npu.synchronize()
    return wall * 1e3, start.elapsed_time(end) / iters


def main():
    art = Path(os.environ.get("KGS_DEBUG_ARTIFACTS", "."))
    art.mkdir(parents=True, exist_ok=True)
    report = {"device": torch.npu.get_device_name(0)}

    for tag, dtype, rows, ncols in [
        ("2048_fp16", torch.float16, 2048, 2048),
        ("1024_fp32", torch.float32, 1024, 1024),
        ("128_fp16", torch.float16, 128, 128),
        ("3d_dim2", torch.float16, 35, 9),
    ]:
        g = torch.Generator().manual_seed(1729)
        inp = torch.randn((rows, ncols), generator=g).to(dtype=dtype, device=DEV)
        src = torch.randn((rows, ncols), generator=g).to(dtype=dtype, device=DEV)
        index = torch.randint(ncols, (rows, ncols), generator=g).to(DEV)
        out = torch.empty_like(inp)
        ref = torch.scatter_add(inp, -1, index, src)
        iters = 20 if rows * ncols >= 1 << 20 else 100
        entry = {"rows": rows, "ncols": ncols, "dtype": str(dtype), "iters": iters}

        for name, blk, ch in [("dense_b128_c32", 128, 32), ("dense_b256_c64", 256, 64),
                              ("dense_b64_c64", 64, 64), ("dense_b128_c128", 128, 128)]:
            blk_eff = min(blk, max(16, 1 << (ncols - 1).bit_length()))
            try:
                grid = (rows, triton.cdiv(ncols, blk_eff))
                fn = lambda: _dense_last[grid](inp, index, src, out, ncols, BLOCK=blk_eff, CHUNK=ch)
                fn()
                torch.npu.synchronize()
                err = float((out.float() - ref.float()).abs().max())
                wall, dev = bench(fn, iters)
                entry[name] = {"block": blk_eff, "chunk": ch, "wall_ms": wall,
                               "device_ms": dev, "max_abs_err": err}
            except Exception as exc:  # noqa: BLE001
                entry[name] = {"block": blk_eff, "chunk": ch, "error": f"{type(exc).__name__}: {exc}"[:400]}
            print(tag, name, json.dumps(entry[name]), flush=True)

        blk = 128
        while blk < ncols and blk < 2048:
            blk *= 2
        grid = (rows, triton.cdiv(ncols, blk))
        wall, dev = bench(lambda: _atomic_last[grid](src, index, out, ncols, BLOCK=blk), iters)
        entry["atomic"] = {"block": blk, "wall_ms": wall, "device_ms": dev}
        wall, dev = bench(lambda: _load_only[grid](inp, index, src, out, ncols, BLOCK=blk), iters)
        entry["load_affine_store"] = {"block": blk, "wall_ms": wall, "device_ms": dev}

        torch.npu.synchronize()
        entry["candidate_max_abs_err"] = float(
            (torch.scatter_add(inp, -1, index, src).float() - ref.float()).abs().max()
        )
        report[tag] = entry
        print(tag, json.dumps(entry), flush=True)

    (art / "probe_dense.json").write_text(json.dumps(report, indent=2))
    print("written", art / "probe_dense.json", flush=True)


if __name__ == "__main__":
    main()
