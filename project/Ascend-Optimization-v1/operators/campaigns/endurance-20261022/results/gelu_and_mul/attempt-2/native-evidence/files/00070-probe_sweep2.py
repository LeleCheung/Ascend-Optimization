"""Bounded diagnostic 3: (a) larger per-program tile areas, (b) host dispatch cost
of a normal triton launch vs a pre-bound CompiledKernel runner.

Read-only probe: no candidate or protected state is modified.
"""

import inspect
import json
import os
import time

import torch
import triton
import triton.language as tl

torch.manual_seed(0)
DEV = "npu"


@triton.jit
def k_tile(
    x_ptr,
    out_ptr,
    half,
    num_col_tiles,
    BLOCK_C: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    row_tile = pid // num_col_tiles
    col_tile = pid - row_tile * num_col_tiles
    rows = row_tile * ROWS + tl.arange(0, ROWS)
    cols = col_tile * BLOCK_C + tl.arange(0, BLOCK_C)
    in_base = rows[:, None] * (2 * half) + cols[None, :]
    gate = tl.load(x_ptr + in_base).to(tl.float32)
    up = tl.load(x_ptr + in_base + half).to(tl.float32)
    inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
    activated = gate * tl.sigmoid(2.0 * inner)
    tl.store(out_ptr + rows[:, None] * half + cols[None, :], activated * up)


@triton.jit
def k_tiny(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=offs < n), mask=offs < n)


def time_call(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    res = {"area_sweep": {}, "dispatch": {}, "api": {}}

    for shape in [(512, 16384), (4096, 16384)]:
        x = torch.randn(shape, dtype=torch.bfloat16, device=DEV)
        half = shape[1] // 2
        out = torch.empty((shape[0], half), dtype=torch.bfloat16, device=DEV)
        key = f"{shape[0]}x{shape[1]}"
        res["area_sweep"][key] = {}
        for bc, rows in [
            (2048, 4),
            (4096, 2),
            (2048, 8),
            (4096, 4),
            (8192, 2),
            (4096, 8),
            (8192, 4),
        ]:
            if bc > half or half % bc or shape[0] % rows:
                continue
            nct = half // bc
            grid = (shape[0] // rows * nct,)
            name = f"bc{bc}_r{rows}_area{bc*rows}"
            try:
                ms = time_call(
                    lambda b=bc, r=rows, g=grid: k_tile[g](x, out, half, nct, b, r, num_warps=4),
                    iters=8 if shape[0] > 512 else 12,
                )
                res["area_sweep"][key][name] = {
                    "ms": ms,
                    "gbs": out.numel() * 2 * 3 / (ms * 1e-3) / 1e9,
                    "grid": grid[0],
                }
            except Exception as exc:  # compile/UB failures recorded, not hidden
                res["area_sweep"][key][name] = {"error": f"{type(exc).__name__}: {exc}"[:400]}
        del x, out

    # host dispatch: normal launch vs pre-bound CompiledKernel runner
    n = 2048
    x = torch.randn(n, dtype=torch.bfloat16, device=DEV)
    out = torch.empty(n, dtype=torch.bfloat16, device=DEV)
    res["dispatch"]["normal_launch_ms"] = time_call(
        lambda: k_tiny[(1,)](x, out, n, 2048, num_warps=4), iters=50
    )
    try:
        kern = k_tiny[(1,)](x, out, n, 2048, num_warps=4)
        res["api"]["returned_type"] = type(kern).__name__
        res["api"]["has_getitem"] = hasattr(kern, "__getitem__")
        runner = kern[(1,)]
        res["api"]["runner_type"] = type(runner).__name__
        try:
            res["api"]["runner_signature"] = str(inspect.signature(runner))
        except Exception as exc:
            res["api"]["runner_signature"] = f"unavailable: {type(exc).__name__}"
        res["api"]["dir_sample"] = [a for a in dir(kern) if not a.startswith("_")][:40]
        # constexpr args are baked in; only runtime args are forwarded
        res["dispatch"]["prebound_runner_ms"] = time_call(
            lambda: runner(x, out, n), iters=50
        )
        res["dispatch"]["prebound_matches"] = bool(
            torch.equal(out, torch.where(torch.arange(n, device=DEV) < n, x, torch.zeros_like(x)))
        )
    except Exception as exc:
        res["api"]["error"] = f"{type(exc).__name__}: {exc}"[:600]

    # does the runner observe a fresh stream? compare two streams' results
    try:
        dev = torch.npu.current_device()
        res["api"]["current_stream"] = str(torch.npu.current_stream(dev))
        res["api"]["stream_api_ok"] = True
    except Exception as exc:
        res["api"]["current_stream"] = f"{type(exc).__name__}: {exc}"

    out_dir = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "probe_sweep2.json"), "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print(json.dumps(res, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
