# SPDX-License-Identifier: Apache-2.0
"""Bounded diagnostic 3 (<=120s): program/transfer structure. Sweeps large BLOCK
values and persistent multi-tile program forms to locate the achievable floor for
a Triton pointwise round on the 4096x4096 workloads, and re-checks small-N.
Writes $KGS_DEBUG_ARTIFACTS/probe3.json incrementally.
"""
import json
import math
import os
import sys
import time
import traceback

import torch
import torch_npu
import triton

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernels as K  # noqa: E402

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = {"env": {}, "timing": {}, "errors": []}


def flush():
    with open(os.path.join(ART, "probe3.json"), "w") as fh:
        json.dump(OUT, fh, indent=1, default=str)


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def flat(kern, x, out, decimals, block):
    n = x.numel()
    kern[((n + block - 1) // block,)](x, out, n, decimals, BLOCK=block)


def loop(kern, x, out, decimals, block, groups):
    n = x.numel()
    iters = max(1, -(-n // (block * groups)))
    kern[(groups,)](x, out, n, decimals, BLOCK=block, ITERS=iters)


def try_time(row, key, fn, iters=20):
    try:
        row[key] = bench(fn, iters=iters)
    except Exception:
        row[key] = "FAIL " + traceback.format_exc()[-140:]


def main():
    props = torch.npu.get_device_properties(torch.npu.current_device())
    OUT["env"] = {"device": props.name,
                  "num_aicore": getattr(props, "num_aicore", None),
                  "num_vectorcore": getattr(props, "num_vectorcore", None),
                  "l2_cache_size": getattr(props, "l2_cache_size", None)}
    flush()

    # ---- large block sweep + persistent forms, fp16 4096x4096 ---------------
    x = torch.randn(4096, 4096, dtype=torch.float32).to(torch.float16).npu()
    out = torch.empty_like(x)
    row = {"torch_round_ms": bench(lambda: torch.round(x, decimals=0, out=out))}
    for block in (8192, 16384, 32768):
        try_time(row, "copy_b%d" % block, lambda b=block: flat(K.k_copy, x, out, 0, b))
        try_time(row, "magic_b%d" % block, lambda b=block: flat(K.k_magic_f32, x, out, 0, b))
    OUT["timing"]["fp16_flat"] = row
    flush()

    row = {}
    for block, groups in ((8192, 64), (8192, 128), (8192, 256), (16384, 64), (16384, 128)):
        try_time(row, "loop_magic_b%d_g%d" % (block, groups),
                 lambda b=block, g=groups: loop(K.k_loop_magic, x, out, 0, b, g))
    try_time(row, "loop_magic_s_b8192_g128", lambda: loop(K.k_loop_magic_s, x, out, 0, 8192, 128))
    try_time(row, "chunk_magic_b8192_g128", lambda: loop(K.k_chunk_magic, x, out, 0, 8192, 128))
    try_time(row, "loop_copy_b8192_g128", lambda: loop(K.k_loop_copy, x, out, 0, 8192, 128))
    OUT["timing"]["fp16_loop"] = row
    flush()

    # ---- best structures on bf16 / fp32 -------------------------------------
    for dtype in (torch.bfloat16, torch.float32):
        tag = str(dtype).split('.')[-1]
        xx = torch.randn(4096, 4096, dtype=torch.float32).to(dtype).npu()
        oo = torch.empty_like(xx)
        row = {"torch_round_ms": bench(lambda: torch.round(xx, decimals=0, out=oo))}
        try_time(row, "magic_b16384", lambda: flat(K.k_magic_f32, xx, oo, 0, 16384))
        try_time(row, "magic_b8192", lambda: flat(K.k_magic_f32, xx, oo, 0, 8192))
        try_time(row, "loop_magic_b8192_g128",
                 lambda: loop(K.k_loop_magic, xx, oo, 0, 8192, 128))
        try_time(row, "copy_b16384", lambda: flat(K.k_copy, xx, oo, 0, 16384))
        OUT["timing"]["4096_" + tag] = row
        flush()

    # ---- small N: per-call walltime of the leading structure ----------------
    for n in (1024, 65536, 1048576):
        xx = torch.randn(n, dtype=torch.float32).to(torch.float16).npu()
        oo = torch.empty_like(xx)
        row = {"torch_round_ms": bench(lambda: torch.round(xx, decimals=0, out=oo), iters=100)}
        try_time(row, "flat_b256", lambda: flat(K.k_magic_f32, xx, oo, 0, 256), iters=100)
        try_time(row, "flat_b4096", lambda: flat(K.k_magic_f32, xx, oo, 0, 4096), iters=100)
        try_time(row, "loop_b8192_g128",
                 lambda: loop(K.k_loop_magic, xx, oo, 0, 8192, 128), iters=100)
        try_time(row, "loop_b8192_g8",
                 lambda: loop(K.k_loop_magic, xx, oo, 0, 8192, 8), iters=100)
        OUT["timing"]["small_%d_fp16" % n] = row
        flush()

    print(json.dumps(OUT, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
