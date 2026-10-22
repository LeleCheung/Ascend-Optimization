"""Decompose the candidate's ~186 us host path to find the serializing call."""

import importlib.util
import json
import os
import time
import traceback

import torch
import torch_npu  # noqa: F401

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
res = {"bench": {}, "notes": []}


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

    from flag_gems.runtime import torch_device_fn
    from flag_gems.utils.random_utils import philox_backend_seed_offset

    x = torch.empty(1048576, dtype=torch.float16, device="npu")
    tiny = torch.empty(1024, dtype=torch.float16, device="npu")
    gen = torch.npu.default_generators[0]

    run_named("00_aten_ref_1M", lambda: torch.ops.aten.uniform(x, 0.0, 1.0))
    run_named("01_full_run_tiny", lambda: m.run(tiny, 0.0, 1.0))
    run_named("02_full_run_1M", lambda: m.run(x, 0.0, 1.0))
    run_named("03_empty_like_1M", lambda: torch.empty_like(x))
    run_named("04_meta_only", lambda: (torch.empty_like(x), x.numel(), x.dtype))
    run_named("05_current_device", lambda: torch_device_fn.current_device())
    run_named("06_initial_seed", lambda: gen.initial_seed())
    run_named("07_philox_seed_offset", lambda: philox_backend_seed_offset(1048576))
    run_named("08_bounds_cached", lambda: m._bounds(torch.float16, 0.0, 1.0))

    # launch with fixed rng and no other host work (pure launch floor for real args)
    grid = (128,)
    args = (x, 1048576, 12345, 4096, 0.0, 0.99951171875, 1.0)
    run_named("10_launch_only_1M", lambda: m._uniform_kernel[grid](
        *args, BLOCK=8192, EVEN=True))
    run_named("11_launch_only_tiny", lambda: m._uniform_kernel[(1,)](
        tiny, 1024, 12345, 4096, 0.0, 0.99951171875, 1.0, BLOCK=1024, EVEN=True))
    run_named("12_launch_plus_empty_1M", lambda: (
        torch.empty_like(x),
        m._uniform_kernel[grid](*args, BLOCK=8192, EVEN=True)))

    # cheap host RNG: cached seed + advancing counter, no device touch
    cnt = [0]
    cache = [-1]

    def rng_counter(n):
        o = cnt[0]
        cnt[0] = o + n + 4
        return 12345, o

    def rng_seedread(n):
        s = gen.initial_seed()
        if s != cache[0]:
            cache[0] = s
            cnt[0] = 0
        o = cnt[0]
        cnt[0] = o + n + 4
        return s & 0x7FFFFFFF, o

    def full_with(rngfn, t, n, grid, block, even):
        out = torch.empty_like(t)
        lo, hi = m._bounds(out.dtype, 0.0, 1.0)
        s, o = rngfn(n)
        m._uniform_kernel[grid](out, n, s, o, lo, hi, 1.0, BLOCK=block, EVEN=even)
        return out

    run_named("20_counter_rng_1M", lambda: full_with(rng_counter, x, 1048576, (128,), 8192, True))
    run_named("21_counter_rng_tiny", lambda: full_with(rng_counter, tiny, 1024, (1,), 1024, True))
    run_named("22_seedread_rng_1M", lambda: full_with(rng_seedread, x, 1048576, (128,), 8192, True))
    run_named("23_seedread_rng_tiny", lambda: full_with(rng_seedread, tiny, 1024, (1,), 1024, True))

    # does calling get_state/set_state between launches serialize?
    def philox_plus_launch():
        out = torch.empty_like(x)
        s, o = philox_backend_seed_offset(1048576)
        m._uniform_kernel[grid](out, 1048576, s & 0x7FFFFFFF, o & 0x7FFFFFFF,
                                0.0, 0.99951171875, 1.0, BLOCK=8192, EVEN=True)
        return out

    run_named("30_philox_plus_launch_1M", philox_plus_launch)

    # argument-count / kwargs sensitivity of the launch
    run_named("40_launch_positional_meta", lambda: m._uniform_kernel[grid](
        x, 1048576, 12345, 4096, 0.0, 0.99951171875, 1.0, 8192, True))

    with open(os.path.join(ART, "decomp.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


main()
