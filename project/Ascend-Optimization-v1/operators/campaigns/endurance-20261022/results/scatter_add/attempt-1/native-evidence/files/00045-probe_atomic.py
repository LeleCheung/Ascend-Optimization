"""Bounded target-side probe: where does the scatter_add time go?

Measures, per call, both host wall time (launch + sync) and device time
(torch.npu.Event) for:
  - the candidate run() on the timed shapes
  - the copy kernel alone (fixed host/launch overhead floor)
  - the atomic last-dim scatter alone
  - a non-atomic scatter store (same memory traffic, no accumulation)
  - the atomic kernel with unique indices (zero conflicts)
  - the atomic kernel with a single repeated index (maximum conflicts)
  - BLOCK variants for the atomic kernel
Outputs tmp/debug/artifacts/probe_atomic.json
"""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl

DEV = "npu:0"


def find_candidate():
    here = Path(__file__).resolve()
    for base in [Path.cwd(), *here.parents]:
        p = base / "tmp" / "main.py"
        if p.is_file():
            return p
    raise SystemExit("main.py not found")


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
def _store_last(src_ptr, idx_ptr, out_ptr, ncols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < ncols
    base = row * ncols
    off = base + col
    idx = tl.load(idx_ptr + off, mask=mask, other=0).to(tl.int32)
    val = tl.load(src_ptr + off, mask=mask, other=0)
    tl.store(out_ptr + (base + idx), val, mask=mask)


@triton.jit(do_not_specialize=["ncols"])
def _gather_last(inp_ptr, idx_ptr, src_ptr, out_ptr, ncols, BLOCK: tl.constexpr):
    """Non-atomic lower bound for one source pass: read index+src, write out."""
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < ncols
    base = row * ncols
    off = base + col
    idx = tl.load(idx_ptr + off, mask=mask, other=0).to(tl.int32)
    val = tl.load(src_ptr + off, mask=mask, other=0)
    tl.store(out_ptr + (base + idx), val, mask=mask)


@triton.jit(do_not_specialize=["n"])
def _copy_flat(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(dst_ptr + off, tl.load(src_ptr + off, mask=mask), mask=mask)


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
    cand_path = find_candidate()
    spec = importlib.util.spec_from_file_location("cand_mod", cand_path)
    cand = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cand)

    report = {"candidate": str(cand_path), "device": torch.npu.get_device_name(0)}

    for tag, dtype, rows, ncols in [
        ("2048_fp16", torch.float16, 2048, 2048),
        ("2048_fp32", torch.float32, 2048, 2048),
        ("128_fp16", torch.float16, 128, 128),
    ]:
        g = torch.Generator().manual_seed(1729)
        inp = torch.randn((rows, ncols), generator=g).to(dtype=dtype, device=DEV)
        src = torch.randn((rows, ncols), generator=g).to(dtype=dtype, device=DEV)
        index = torch.randint(ncols, (rows, ncols), generator=g).to(DEV)
        out = torch.empty_like(inp)
        n = rows * ncols
        block = 128
        while block < ncols and block < 2048:
            block *= 2
        grid = (rows, triton.cdiv(ncols, block))
        iters = 20 if n >= 1 << 20 else 100
        entry = {"rows": rows, "ncols": ncols, "dtype": str(dtype), "block": block, "iters": iters}

        wall, dev = bench(lambda: _copy_flat[(triton.cdiv(n, 8192),)](inp, out, n, BLOCK=8192), iters)
        entry["copy_only"] = {"wall_ms": wall, "device_ms": dev}

        wall, dev = bench(
            lambda: _atomic_last[grid](src, index, out, ncols, BLOCK=block), iters
        )
        entry["atomic_random_idx"] = {"wall_ms": wall, "device_ms": dev}

        wall, dev = bench(lambda: _store_last[grid](src, index, out, ncols, BLOCK=block), iters)
        entry["store_random_idx"] = {"wall_ms": wall, "device_ms": dev}

        uniq = (
            torch.arange(ncols, device=DEV, dtype=torch.int64)
            .unsqueeze(0)
            .expand(rows, ncols)
            .contiguous()
        )
        wall, dev = bench(lambda: _atomic_last[grid](src, uniq, out, ncols, BLOCK=block), iters)
        entry["atomic_unique_idx"] = {"wall_ms": wall, "device_ms": dev}

        same = torch.zeros((rows, ncols), dtype=torch.int64, device=DEV)
        wall, dev = bench(lambda: _atomic_last[grid](src, same, out, ncols, BLOCK=block), iters)
        entry["atomic_same_idx"] = {"wall_ms": wall, "device_ms": dev}

        for alt in (256, 512, 1024):
            if alt > block:
                continue
            g2 = (rows, triton.cdiv(ncols, alt))
            wall, dev = bench(lambda: _atomic_last[g2](src, index, out, ncols, BLOCK=alt), iters)
            entry[f"atomic_block_{alt}"] = {"wall_ms": wall, "device_ms": dev}

        wall, dev = bench(lambda: cand.run(inp, -1, index, src), iters)
        entry["candidate_run"] = {"wall_ms": wall, "device_ms": dev}

        # correctness of the probe kernels vs reference (small slices only)
        if n <= 1 << 20:
            ref = torch.scatter_add(inp, -1, index, src)
            probe = cand.run(inp, -1, index, src)
            entry["candidate_max_abs_err"] = float((probe.float() - ref.float()).abs().max())
            torch.npu.synchronize()
        report[tag] = entry
        print(tag, json.dumps(entry), flush=True)

    (art / "probe_atomic.json").write_text(json.dumps(report, indent=2))
    print("written", art / "probe_atomic.json", flush=True)


if __name__ == "__main__":
    main()
