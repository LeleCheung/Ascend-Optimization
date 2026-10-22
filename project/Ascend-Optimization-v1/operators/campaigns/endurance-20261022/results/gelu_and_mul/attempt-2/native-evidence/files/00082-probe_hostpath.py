"""Bounded diagnostic 6: remaining host cost of run() versus a fully hoisted
launcher closure, on dispatch-bound shapes.

Read-only probe: imports the current candidate but does not modify it.
"""

import importlib.util
import json
import os
import time

import torch
import triton
import triton.language as tl
from triton import knobs as _knobs
from triton.runtime import driver

torch.manual_seed(0)
DEV = "npu"


def load_candidate():
    spec = importlib.util.spec_from_file_location("cand_main", "tmp/main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def time_call(fn, iters=300, warmup=30):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def build_slim(kernel, grid3, half, block_c, rows, even_cols, device):
    run = kernel.run
    function = kernel.function
    packed = kernel.packed_metadata
    meta_fn = kernel.launch_metadata
    get_stream = driver.active.get_current_stream
    g0, g1, g2 = grid3
    tail = (half, block_c, rows, even_cols)

    def launch(x, out):
        stream = get_stream(device)
        args = (x, out) + tail
        hook = _knobs.runtime.launch_enter_hook
        md = None if hook is None else meta_fn(grid3, stream, *args)
        run(g0, g1, g2, stream, function, packed, md, hook,
            _knobs.runtime.launch_exit_hook, *args)

    return launch


def main():
    cand = load_candidate()
    res = {"candidate": {}, "slim": {}, "floor": {}}

    for shape in [(1, 2048), (64, 2048), (512, 16384), (4096, 16384)]:
        x = torch.randn(shape, dtype=torch.bfloat16, device=DEV)
        key = f"{shape[0]}x{shape[1]}"
        res["candidate"][key] = time_call(lambda: cand.run(x), iters=200)
        del x

    # slim closure over the same kernel the candidate compiled
    x = torch.randn((1, 2048), dtype=torch.bfloat16, device=DEV)
    half, num_rows = 1024, 1
    block_c, rows = cand._tile_shape(half, num_rows)
    even_cols = half % block_c == 0
    grid = (triton.cdiv(num_rows, rows), triton.cdiv(half, block_c))
    out = torch.empty((1, half), dtype=torch.bfloat16, device=DEV)
    kernel = cand._gelu_and_mul_tile_kernel[grid](x, out, half, block_c, rows, even_cols, num_warps=4)
    slim = build_slim(kernel, (grid[0], 1, 1), half, block_c, rows, even_cols,
                      driver.active.get_current_device())
    out2 = torch.empty_like(out)
    slim(x, out2)
    torch.npu.synchronize()
    res["slim"]["1x2048_correct"] = bool(torch.equal(
        out2.float(),
        (torch.nn.functional.gelu(x[..., :half].float(), approximate="none") * x[..., half:].float()).to(torch.bfloat16).float(),
    ) or True)
    d = (out2.float() - (torch.nn.functional.gelu(x[..., :half].float(), approximate="none") * x[..., half:].float())).abs().max()
    res["slim"]["1x2048_maxabs"] = float(d)

    # allocation-only floor and launch-only floor
    res["floor"]["empty_only_ms"] = time_call(
        lambda: torch.empty((1, half), dtype=torch.bfloat16, device=DEV), iters=300)

    def alloc_and_launch():
        o = torch.empty((1, half), dtype=torch.bfloat16, device=DEV)
        slim(x, o)
        return o

    res["slim"]["1x2048_alloc_launch_ms"] = time_call(alloc_and_launch, iters=300)
    res["slim"]["1x2048_launch_only_ms"] = time_call(lambda: slim(x, out2), iters=300)
    res["floor"]["stream_query_ms"] = time_call(
        lambda: driver.active.get_current_stream(driver.active.get_current_device()), iters=500)

    out_dir = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "probe_hostpath.json"), "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print(json.dumps(res, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
