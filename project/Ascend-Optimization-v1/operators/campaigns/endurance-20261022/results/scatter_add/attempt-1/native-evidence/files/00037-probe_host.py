"""Bounded probe #4: host-side cost of run() on the small (128^2) family.

Splits the per-call wall time into allocation, dispatch-decision and launch
cost, and verifies a trimmed host path against torch.scatter_add.
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
def _dense(inp_ptr, idx_ptr, src_ptr, out_ptr, ncols, BLOCK: tl.constexpr, CHUNK: tl.constexpr):
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


def run_trimmed(inp, dim, index, src):
    ndim = inp.ndim
    if dim < 0:
        dim += ndim
    n = index.numel()
    if n and dim == ndim - 1:
        ncols = inp.shape[-1]
        if (
            index.shape == inp.shape
            and src.shape == index.shape
            and inp.is_contiguous()
            and index.is_contiguous()
            and src.is_contiguous()
            and ncols >= 32
            and n * ncols < 903 * n + 10770000
        ):
            out = torch.empty_like(inp)
            _dense[(n // ncols, (ncols + 127) // 128)](
                inp, index, src, out, ncols, BLOCK=128, CHUNK=32
            )
            return out
    return None


def bench(fn, iters):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    art = Path(os.environ.get("KGS_DEBUG_ARTIFACTS", "."))
    art.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve()
    cand = None
    for base in [Path.cwd(), *here.parents]:
        p = base / "tmp" / "main.py"
        if p.is_file():
            spec = importlib.util.spec_from_file_location("cand_mod", p)
            cand = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cand)
            break
    report = {}

    for tag, dtype, rows, ncols in [("128_fp16", torch.float16, 128, 128),
                                   ("128_fp32", torch.float32, 128, 128)]:
        g = torch.Generator().manual_seed(1729)
        inp = torch.randn((rows, ncols), generator=g).to(dtype=dtype, device=DEV)
        src = torch.randn((rows, ncols), generator=g).to(dtype=dtype, device=DEV)
        index = torch.randint(ncols, (rows, ncols), generator=g).to(DEV)
        ref = torch.scatter_add(inp, -1, index, src)
        out = torch.empty_like(inp)
        n = rows * ncols
        e = {}

        e["empty_shape_kwargs_ms"] = bench(
            lambda: torch.empty(inp.shape, dtype=inp.dtype, device=inp.device), 2000)
        e["empty_like_ms"] = bench(lambda: torch.empty_like(inp), 2000)
        e["launch_only_ms"] = bench(
            lambda: _dense[(rows, (ncols + 127) // 128)](inp, index, src, out, ncols,
                                                         BLOCK=128, CHUNK=32), 200)
        got = run_trimmed(inp, -1, index, src)
        torch.npu.synchronize()
        e["trimmed_max_abs_err"] = float((got.float() - ref.float()).abs().max())
        e["trimmed_ms"] = bench(lambda: run_trimmed(inp, -1, index, src), 200)
        if cand is not None:
            got2 = cand.run(inp, -1, index, src)
            torch.npu.synchronize()
            e["current_max_abs_err"] = float((got2.float() - ref.float()).abs().max())
            e["current_ms"] = bench(lambda: cand.run(inp, -1, index, src), 200)
        report[tag] = e
        print(tag, json.dumps(e), flush=True)

    (art / "probe_host.json").write_text(json.dumps(report, indent=2))
    print("written", flush=True)


if __name__ == "__main__":
    main()
