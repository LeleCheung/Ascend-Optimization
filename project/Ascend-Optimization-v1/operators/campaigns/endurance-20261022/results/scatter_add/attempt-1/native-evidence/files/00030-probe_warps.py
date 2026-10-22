"""Bounded probe #3: launch-shape levers for the data-dependent atomic path
(mask removal, num_warps) and dense single-kernel variants for small rows.

Everything is verified against torch.scatter_add on the same inputs.
"""

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
def _atomic_masked(src_ptr, idx_ptr, out_ptr, ncols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < ncols
    base = row * ncols
    off = base + col
    idx = tl.load(idx_ptr + off, mask=mask, other=0).to(tl.int32)
    val = tl.load(src_ptr + off, mask=mask, other=0)
    tl.atomic_add(out_ptr + (base + idx), val, mask=mask, sem="relaxed")


@triton.jit(do_not_specialize=["ncols"])
def _atomic_nomask(src_ptr, idx_ptr, out_ptr, ncols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    base = row * ncols
    off = base + col
    idx = tl.load(idx_ptr + off).to(tl.int32)
    val = tl.load(src_ptr + off)
    tl.atomic_add(out_ptr + (base + idx), val, sem="relaxed")


@triton.jit(do_not_specialize=["ncols"])
def _dense_small(inp_ptr, idx_ptr, src_ptr, out_ptr, ncols, BLOCK: tl.constexpr, CHUNK: tl.constexpr):
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
        ("1024_fp16", torch.float16, 1024, 1024),
        ("128_fp16", torch.float16, 128, 128),
    ]:
        g = torch.Generator().manual_seed(1729)
        inp = torch.randn((rows, ncols), generator=g).to(dtype=dtype, device=DEV)
        src = torch.randn((rows, ncols), generator=g).to(dtype=dtype, device=DEV)
        index = torch.randint(ncols, (rows, ncols), generator=g).to(DEV)
        out = torch.empty_like(inp)
        ref = torch.scatter_add(inp, -1, index, src)
        iters = 20 if rows * ncols >= 1 << 20 else 100
        entry = {"rows": rows, "ncols": ncols, "iters": iters}

        blk = 128
        while blk < ncols and blk < 2048:
            blk *= 2
        grid = (rows, triton.cdiv(ncols, blk))

        for warps in (1, 4, 8, 16, 32):
            for masked in (True, False):
                name = f"atomic_w{warps}_{'mask' if masked else 'nomask'}"
                kern = _atomic_masked if masked else _atomic_nomask
                try:
                    def fn(kern=kern, warps=warps):
                        kern[grid](src, index, out, ncols, BLOCK=blk, num_warps=warps)
                    out.zero_()
                    fn()
                    torch.npu.synchronize()
                    err = float((out.float() - ref.float()).abs().max())
                    wall, dev = bench(fn, iters)
                    entry[name] = {"wall_ms": wall, "device_ms": dev, "max_abs_err": err}
                except Exception as exc:  # noqa: BLE001
                    entry[name] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
                print(tag, name, json.dumps(entry[name]), flush=True)

        # dense single-kernel only for small rows*ncols work
        if rows * ncols * ncols <= 4 * (1 << 20):
            for dblk, ch in [(16, 16), (32, 32), (64, 64), (128, 32)]:
                dblk_eff = min(dblk, max(16, 1 << (ncols - 1).bit_length()))
                name = f"dense_b{dblk_eff}_c{ch}"
                try:
                    ggrid = (rows, triton.cdiv(ncols, dblk_eff))
                    def fn(dblk_eff=dblk_eff, ch=ch, ggrid=ggrid):
                        _dense_small[ggrid](inp, index, src, out, ncols, BLOCK=dblk_eff, CHUNK=ch)
                    fn()
                    torch.npu.synchronize()
                    err = float((out.float() - ref.float()).abs().max())
                    wall, dev = bench(fn, iters)
                    entry[name] = {"wall_ms": wall, "device_ms": dev, "max_abs_err": err}
                except Exception as exc:  # noqa: BLE001
                    entry[name] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
                print(tag, name, json.dumps(entry[name]), flush=True)

        report[tag] = entry
    (art / "probe_warps.json").write_text(json.dumps(report, indent=2))
    print("written", flush=True)


if __name__ == "__main__":
    main()
