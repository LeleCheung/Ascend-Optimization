"""Bounded diagnostic 5: measure the pre-bound launcher fast path (mirroring the
installed JITFunction.run launch, minus binder/cache-key work).

Read-only probe.
"""

import json
import os
import time

import torch
import triton
import triton.language as tl
from triton import knobs
from triton.runtime import driver

torch.manual_seed(0)
DEV = "npu"


@triton.jit
def k_tiny(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m), mask=m)


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
    tl.store(out_ptr + rows[:, None] * half + cols[None, :], gate * tl.sigmoid(2.0 * inner) * up)


def time_call(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


class FastLaunch:
    """Exact mirror of JITFunction.run's launch, with binding/key work cached."""

    __slots__ = ("kernel", "grid", "args", "device", "md_cached")

    def __init__(self, kernel, grid, args):
        self.kernel = kernel
        self.grid = (grid[0], grid[1] if len(grid) > 1 else 1, grid[2] if len(grid) > 2 else 1)
        self.args = args
        self.device = driver.active.get_current_device()

    def __call__(self):
        k = self.kernel
        stream = driver.active.get_current_stream(self.device)
        md = k.launch_metadata(self.grid, stream, *self.args)
        k.run(
            self.grid[0], self.grid[1], self.grid[2], stream,
            k.function, k.packed_metadata, md,
            knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook,
            *self.args,
        )

    def call_static_md(self):
        k = self.kernel
        stream = driver.active.get_current_stream(self.device)
        k.run(
            self.grid[0], self.grid[1], self.grid[2], stream,
            k.function, k.packed_metadata, self.md_cached,
            knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook,
            *self.args,
        )


def main():
    res = {}
    n = 2048
    x = torch.randn(n, dtype=torch.bfloat16, device=DEV)
    out = torch.empty(n, dtype=torch.bfloat16, device=DEV)
    g = (1, 1, 1)
    args = (x, out, n, 2048)

    res["normal_launch_ms"] = time_call(lambda: k_tiny[g](x, out, n, 2048, num_warps=4))
    kernel = k_tiny[g](x, out, n, 2048, num_warps=4)
    res["kernel_type"] = type(kernel).__name__
    fast = FastLaunch(kernel, g, args)
    out.zero_()
    fast()
    torch.npu.synchronize()
    res["fast_correct"] = bool(torch.equal(out, x))
    res["fast_launch_ms"] = time_call(fast)

    x2 = torch.randn(n, dtype=torch.bfloat16, device=DEV)
    out2 = torch.empty(n, dtype=torch.bfloat16, device=DEV)
    fast2 = FastLaunch(kernel, g, (x2, out2, n, 2048))
    fast2()
    torch.npu.synchronize()
    res["fast_new_tensors_correct"] = bool(torch.equal(out2, x2))

    # static launch_metadata variant
    stream = driver.active.get_current_stream(driver.active.get_current_device())
    fast.md_cached = kernel.launch_metadata(fast.grid, stream, *args)
    res["static_md_launch_ms"] = time_call(fast.call_static_md)

    # real workload shapes
    res["tile"] = {}
    for shape in [(1, 2048), (64, 2048), (512, 16384), (4096, 16384)]:
        half = shape[1] // 2
        xx = torch.randn(shape, dtype=torch.bfloat16, device=DEV)
        oo = torch.empty((shape[0], half), dtype=torch.bfloat16, device=DEV)
        bc, rows = (1024, 2) if half <= 1024 else (4096, 2)
        nct = half // bc
        gt = (shape[0] // rows * nct, 1, 1)
        a = (xx, oo, half, nct, bc, rows)
        key = f"{shape[0]}x{shape[1]}"
        it = 100
        normal = time_call(lambda: k_tile[gt](xx, oo, half, nct, bc, rows, num_warps=4), iters=it)
        kk = k_tile[gt](xx, oo, half, nct, bc, rows, num_warps=4)
        ff = FastLaunch(kk, gt, a)
        ff()
        torch.npu.synchronize()
        fastms = time_call(ff, iters=it)
        # correctness against an independent triton computation
        ref = torch.nn.functional.gelu(xx[..., :half].float(), approximate="none") * xx[..., half:].float()
        ref = ref.to(torch.bfloat16)
        d = (oo.float() - ref.float()).abs()
        res["tile"][key] = {
            "normal_ms": normal,
            "fast_ms": fastms,
            "grid": gt[0],
            "maxabs": float(d.max()),
            "num_gt_1p5e-2": int((d > 0.015).sum()),
        }
        del xx, oo, ref, d

    out_dir = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "probe_fastlaunch.json"), "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print(json.dumps(res, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
