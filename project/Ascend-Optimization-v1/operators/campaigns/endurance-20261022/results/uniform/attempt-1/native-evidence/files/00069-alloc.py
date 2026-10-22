"""Find the cheapest allocation + launch combination for the uniform host path."""

import importlib.util
import json
import os
import time
import traceback

import torch
import torch_npu  # noqa: F401

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
res = {"bench": {}}


def bench(fn, iters=80, warmup=20, trials=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(trials):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) / iters)
    ts.sort()
    return round(ts[len(ts) // 2] * 1e6, 2)


def run_named(name, fn):
    try:
        res["bench"][name] = bench(fn)
    except Exception:
        res["bench"][name] = "ERR:" + traceback.format_exc().splitlines()[-1][:180]


def main():
    spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    gen = torch.npu.default_generators[0]
    x = torch.empty(1048576, dtype=torch.float16, device="npu")
    tiny = torch.empty(1024, dtype=torch.float16, device="npu")
    grid = (128,)
    cnt = [0]

    def kern(out, n, g):
        o = cnt[0]
        cnt[0] = o + n + 4
        g[0](out, n, 12345, o, 0.0, 0.99951171875, 1.0, 8192, True)

    def alloc_like():
        out = torch.empty_like(x)
        kern(out, 1048576, grid)

    def alloc_like_cf():
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        kern(out, 1048576, grid)

    def alloc_empty():
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        kern(out, 1048576, grid)

    def alloc_new_empty():
        out = x.new_empty(x.shape)
        kern(out, 1048576, grid)

    def alloc_empty_1d():
        out = torch.empty(1048576, dtype=x.dtype, device=x.device)
        kern(out, 1048576, grid)

    run_named("aten_ref_1M", lambda: torch.ops.aten.uniform(x, 0.0, 1.0))
    run_named("launch_only_1M", lambda: kern(torch.empty(1, device="npu"), 0, grid))
    run_named("like_1M", alloc_like)
    run_named("like_cf_1M", alloc_like_cf)
    run_named("empty_1M", alloc_empty)
    run_named("new_empty_1M", alloc_new_empty)
    run_named("empty_1d_1M", alloc_empty_1d)
    run_named("like_tiny", lambda: (kern(torch.empty_like(tiny), 1024, (1,))))
    run_named("empty_1d_tiny", lambda: (
        kern(torch.empty(1024, dtype=tiny.dtype, device=tiny.device), 1024, (1,))))

    # seed read cost inside the real path
    def with_seedread():
        s = gen.initial_seed()
        out = torch.empty_like(x)
        kern(out, 1048576, grid)

    run_named("like_plus_seedread_1M", with_seedread)

    with open(os.path.join(ART, "alloc.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


main()
