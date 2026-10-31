"""Tiling sweep through the candidate's own direct-launch plumbing, so the
~28 us host floor is identical for every variant and device-time differences
are visible.  Diagnostics only.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main as cand  # noqa: E402

ART = Path(os.environ.get("KGS_DEBUG_ARTIFACTS", ".")) / "probe4.json"
DEV = "npu:0"
out = {"sweep": {}, "props": {}}


@triton.jit
def _neg_gs(X, BLOCK: tl.constexpr, ITERS: tl.constexpr, STEP: tl.constexpr):
    base = tl.program_id(0) * BLOCK
    for i in tl.static_range(ITERS):
        o = base + i * STEP + tl.arange(0, BLOCK)
        tl.store(X + o, -tl.load(X + o))


def launch_exact(t, block, grid, tag):
    key = ("probe4e", t.dtype, block, tag)
    if not cand._direct_launch(key, cand._neg_exact, t, (), ("BLOCK",), (block,), grid):
        cand._neg_exact[(grid,)](t, BLOCK=block)


def launch_gs(t, block, iters, grid, tag):
    step = grid * block
    key = ("probe4g", t.dtype, block, iters, tag)
    if not cand._direct_launch(key, _neg_gs, t, (), ("BLOCK", "ITERS", "STEP"),
                               (block, iters, step), grid):
        _neg_gs[(grid,)](t, BLOCK=block, ITERS=iters, STEP=step)


def bench(fn, iters=80, warmup=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(3):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) / iters * 1e3)
    ts.sort()
    return round(ts[1], 5)


def run_cfg(t, name, fn):
    try:
        vals = []
        for _ in range(2):          # repeat the whole measurement for stability
            vals.append(bench(fn))
        out["sweep"].setdefault(name, {})[t.dtype.__str__().replace("torch.", "")] = min(vals)
    except Exception:
        out["sweep"].setdefault(name, {})[t.dtype.__str__().replace("torch.", "")] = \
            "ERR:" + traceback.format_exc().strip().splitlines()[-1][:160]


def main():
    from triton.runtime import driver as tdrv
    out["props"] = dict(tdrv.active.utils.get_device_properties(torch.npu.current_device()))
    nvec = out["props"]["num_vectorcore"]

    big = {}
    for dt in (torch.float16, torch.float32):
        big[dt] = torch.zeros((4096, 4096), dtype=dt, device=DEV)
    mid = {torch.float32: torch.zeros(1048576, dtype=torch.float32, device=DEV)}
    small = {torch.float32: torch.zeros(65536, dtype=torch.float32, device=DEV)}

    for dt, t in big.items():
        n = t.numel()
        for block in (4096, 8192, 16384, 32768):
            g = (n + block - 1) // block
            run_cfg(t, f"big_exact_b{block}_g{g}", lambda t=t, b=block, g=g: launch_exact(t, b, g, 0))
        for iters in (4, 8):
            block = 16384
            g = (n + block * iters - 1) // (block * iters)
            run_cfg(t, f"big_gs_b{block}_i{iters}_g{g}",
                    lambda t=t, b=block, i=iters, g=g: launch_gs(t, b, i, g, 0))
        run_cfg(t, "big_seed_tiling", lambda t=t: cand.run(t))
        run_cfg(t, "torch_neg_", lambda t=t: t.neg_())

    for dt, t in list(mid.items()) + list(small.items()):
        n = t.numel()
        for block in (4096, 8192, 16384):
            g = (n + block - 1) // block
            run_cfg(t, f"n{n}_exact_b{block}_g{g}", lambda t=t, b=block, g=g: launch_exact(t, b, g, 0))
        block, iters = 8192, 4
        g = (n + block * iters - 1) // (block * iters)
        run_cfg(t, f"n{n}_gs_b{block}_i{iters}_g{g}",
                lambda t=t, b=block, i=iters, g=g: launch_gs(t, b, i, g, 0))
        run_cfg(t, "candidate_run", lambda t=t: cand.run(t))

    out["nvec"] = nvec


try:
    main()
except Exception:
    out["fatal"] = traceback.format_exc()
finally:
    ART.parent.mkdir(parents=True, exist_ok=True)
    ART.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str)[:8000])
