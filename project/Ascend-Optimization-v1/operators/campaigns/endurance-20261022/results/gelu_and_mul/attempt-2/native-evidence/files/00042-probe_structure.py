"""Bounded diagnostic 2: find the achievable structure for gelu_and_mul on Ascend.

Compares contiguous per-row 2D tiling (no per-lane div/mod) against a pure copy
kernel (bandwidth ceiling) across BLOCK/ROWS/num_warps, and verifies the
tanh-sigmoid GELU formula against the NPU reference numerically.
"""

import json
import os
import time

import torch
import triton
import triton.language as tl

torch.manual_seed(0)
DEV = "npu"


@triton.jit
def k_copy(x_ptr, out_ptr, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=offs < total), mask=offs < total)


@triton.jit
def k_tile(
    x_ptr,
    out_ptr,
    half,
    num_col_tiles,
    BLOCK_C: tl.constexpr,
    ROWS: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_tile = pid // num_col_tiles
    col_tile = pid - row_tile * num_col_tiles
    rows = row_tile * ROWS + tl.arange(0, ROWS)
    cols = col_tile * BLOCK_C + tl.arange(0, BLOCK_C)
    in_base = rows[:, None] * (2 * half) + cols[None, :]
    gate = tl.load(x_ptr + in_base).to(tl.float32)
    up = tl.load(x_ptr + in_base + half).to(tl.float32)

    if MODE == 0:
        activated = gate
    elif MODE == 1:
        activated = gate * (0.5 * (1.0 + tl.erf(gate * 0.7071067811865475)))
    else:
        inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
        activated = gate * tl.sigmoid(2.0 * inner)

    tl.store(out_ptr + rows[:, None] * half + cols[None, :], activated * up)


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
    res = {"copy": {}, "tile": {}, "numerics": {}}

    # bandwidth ceiling: pure copy of an output-sized bf16 buffer
    for shape in [(512, 16384), (4096, 16384)]:
        n = shape[0] * shape[1] // 2
        src = torch.randn(n, dtype=torch.bfloat16, device=DEV)
        dst = torch.empty(n, dtype=torch.bfloat16, device=DEV)
        for blk in (1024, 4096):
            grid = (triton.cdiv(n, blk),)
            ms = time_call(lambda b=blk, g=grid: k_copy[g](src, dst, n, b, num_warps=4))
            gbs = (n * 2 * 2) / (ms * 1e-3) / 1e9
            res["copy"][f"{shape[0]}x{shape[1]}_blk{blk}"] = {"ms": ms, "gbs": gbs}
        del src, dst

    # structure sweep
    for shape in [(1, 2048), (64, 16384), (512, 16384), (4096, 16384)]:
        x = torch.randn(shape, dtype=torch.bfloat16, device=DEV)
        half = shape[1] // 2
        out = torch.empty((shape[0], half), dtype=torch.bfloat16, device=DEV)
        key = f"{shape[0]}x{shape[1]}"
        res["tile"][key] = {}
        for bc, rows, warps in [
            (1024, 1, 4),
            (2048, 1, 4),
            (4096, 1, 4),
            (8192, 1, 4),
            (2048, 1, 2),
            (4096, 1, 2),
            (2048, 4, 4),
            (4096, 2, 4),
            (1024, 4, 4),
        ]:
            if bc > half or half % bc != 0 or shape[0] % rows != 0:
                continue
            nct = half // bc
            grid = (shape[0] // rows * nct,)
            if grid[0] > 200000:
                continue
            iters = 10 if shape[0] <= 512 else 6
            ms = time_call(
                lambda b=bc, r=rows, w=warps, g=grid: k_tile[g](
                    x, out, half, nct, b, r, 1, num_warps=w
                ),
                iters=iters,
            )
            gbs = (out.numel() * 2 * 3) / (ms * 1e-3) / 1e9
            res["tile"][key][f"bc{bc}_r{rows}_w{warps}"] = {"ms": ms, "gbs": gbs}
        del x, out

    # numerics: tanh-sigmoid formula vs NPU reference (diagnostic only)
    x = torch.randn((4096, 16384), dtype=torch.bfloat16, device=DEV)
    half = x.shape[1] // 2
    out = torch.empty((x.shape[0], half), dtype=torch.bfloat16, device=DEV)
    nct = half // 4096
    k_tile[(x.shape[0] * nct,)](x, out, half, nct, 4096, 1, 2, num_warps=4)
    ref = (
        torch.nn.functional.gelu(x[..., :half].float(), approximate="none")
        * x[..., half:].float()
    ).to(torch.bfloat16)
    diff = (out.float() - ref.float()).abs()
    res["numerics"]["tanh_vs_npu_ref_bf16"] = {
        "maxabs": float(diff.max()),
        "meanabs": float(diff.mean()),
        "num_gt_tol": int((diff > 0.015).sum()),
        "total": int(diff.numel()),
    }
    xf = torch.randn(1 << 20, dtype=torch.float32, device=DEV)
    ref_f = torch.nn.functional.gelu(xf, approximate="none")
    hf = xf.numel()
    gate = xf[: hf // 2].contiguous()
    upf = xf[hf // 2:].contiguous()
    inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
    mine = gate * torch.sigmoid(2.0 * inner)
    refg = torch.nn.functional.gelu(gate, approximate="none")
    gd = (mine - refg).abs()
    res["numerics"]["tanh_vs_npu_ref_fp32"] = {
        "maxabs": float(gd.max()),
        "num_gt_1e-4": int((gd > 1e-4).sum()),
    }
    del x, out, ref, diff, xf, ref_f

    out_dir = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "probe_structure.json"), "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print(json.dumps(res, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
