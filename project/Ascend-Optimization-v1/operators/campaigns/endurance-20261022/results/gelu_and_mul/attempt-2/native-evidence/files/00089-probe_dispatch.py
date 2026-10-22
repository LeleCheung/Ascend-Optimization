"""Bounded diagnostic 4: how much host dispatch does a pre-bound CompiledKernel
runner save versus a normal `kernel[grid](...)` launch on this FlagTree build?

Read-only probe.
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
def k_tiny(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=offs < n), mask=offs < n)


@triton.jit
def k_tile(x_ptr, out_ptr, half, num_col_tiles, BLOCK_C: tl.constexpr, ROWS: tl.constexpr):
    pid = tl.program_id(0)
    row_tile = pid // num_col_tiles
    col_tile = pid - row_tile * num_col_tiles
    rows = row_tile * ROWS + tl.arange(0, ROWS)
    cols = col_tile * BLOCK_C + tl.arange(0, BLOCK_C)
    in_base = rows[:, None] * (2 * half) + cols[None, :]
    gate = tl.load(x_ptr + in_base).to(tl.float32)
    up = tl.load(x_ptr + in_base + half).to(tl.float32)
    inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
    tl.store(
        out_ptr + rows[:, None] * half + cols[None, :],
        gate * tl.sigmoid(2.0 * inner) * up,
    )


def time_call(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    res = {"tiny": {}, "tile": {}}

    n = 2048
    x = torch.randn(n, dtype=torch.bfloat16, device=DEV)
    out = torch.empty(n, dtype=torch.bfloat16, device=DEV)
    grid3 = (1, 1, 1)

    res["tiny"]["normal_launch_ms"] = time_call(lambda: k_tiny[grid3](x, out, n, 2048, num_warps=4))

    kern = k_tiny[grid3](x, out, n, 2048, num_warps=4)
    runner = kern[grid3]
    out.zero_()
    runner(x, out, n)
    torch.npu.synchronize()
    res["tiny"]["prebound_correct"] = bool(torch.equal(out, x))
    res["tiny"]["prebound_runner_ms"] = time_call(lambda: runner(x, out, n))

    # fresh tensors through the same pre-bound runner (no per-call rebinding)
    x2 = torch.randn(n, dtype=torch.bfloat16, device=DEV)
    out2 = torch.empty(n, dtype=torch.bfloat16, device=DEV)
    runner(x2, out2, n)
    torch.npu.synchronize()
    res["tiny"]["prebound_new_tensors_correct"] = bool(torch.equal(out2, x2))

    # per-call stream read (what a safe cached launcher must do)
    stream = torch.npu.current_stream(DEV)
    res["tiny"]["with_explicit_stream_ms"] = time_call(lambda: runner(x, out, n, stream=stream))

    # realistic case: does the saving survive at a real workload shape?
    for shape in [(1, 2048), (64, 2048), (512, 16384)]:
        half = shape[1] // 2
        xx = torch.randn(shape, dtype=torch.bfloat16, device=DEV)
        oo = torch.empty((shape[0], half), dtype=torch.bfloat16, device=DEV)
        bc, rows = (1024, 2) if half <= 1024 else (4096, 2)
        nct = half // bc
        g = (shape[0] // rows * nct, 1, 1)
        key = f"{shape[0]}x{shape[1]}"
        normal = time_call(lambda: k_tile[g](xx, oo, half, nct, bc, rows, num_warps=4), iters=100)
        kk = k_tile[g](xx, oo, half, nct, bc, rows, num_warps=4)
        rr = kk[g]
        fast = time_call(lambda: rr(xx, oo, half, nct), iters=100)
        res["tile"][key] = {"normal_ms": normal, "prebound_ms": fast, "grid": g[0]}
        del xx, oo

    out_dir = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "probe_dispatch.json"), "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print(json.dumps(res, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
