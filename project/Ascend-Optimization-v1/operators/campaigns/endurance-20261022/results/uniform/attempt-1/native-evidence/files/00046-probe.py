"""Bounded micro-benchmark on the target: attribute the seed's ~0.35 ms/call.

Same methodology as the evaluator's walltime mode: loop of calls, one
synchronize at the end, median of 3 trials.  All kernels here read the RNG
state that was fetched once, so the numbers separate host cost, philox cost
and store cost.
"""

import importlib.util
import json
import os
import time
import traceback

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
res = {"variants": {}, "notes": []}


def bench(fn, iters=60, warmup=15, trials=3):
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
    return ts[len(ts) // 2] * 1e6  # microseconds per call


@triton.jit
def u2f(x):
    x = x.to(tl.int32, bitcast=True)
    x = tl.where(x < 0, -x - 1, x)
    return x.to(tl.float32) * 4.6566127342e-10


@triton.jit
def k_fill(o, N, val, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.full((BLOCK,), 1.0, tl.float32) * val
    if EVEN:
        tl.store(o + i, v)
    else:
        tl.store(o + i, v, mask=i < N)


@triton.jit
def k_hash(o, N, seed, off, lo, scale, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = (i + off).to(tl.uint32) * tl.full((BLOCK,), 0x9E3779B1, tl.uint32)
    x = x ^ (x >> 15)
    x = x * tl.full((BLOCK,), 0x85EBCA6B, tl.uint32)
    x = x ^ (x >> 13)
    u = u2f(x)
    v = u * scale + lo
    if EVEN:
        tl.store(o + i, v)
    else:
        tl.store(o + i, v, mask=i < N)


@triton.jit
def k_uni4(o, N, seed, poff, lo, hi, scale, BLOCK: tl.constexpr, EVEN: tl.constexpr,
           CLAMP: tl.constexpr):
    seed = seed.to(tl.int64)
    poff = poff.to(tl.int64)
    c0 = (poff & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((poff >> 32) & 0xFFFFFFFF).to(tl.uint32)
    lane = tl.arange(0, BLOCK)
    c0 += tl.program_id(0) * BLOCK + lane
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(seed, c0, c1, _O, _O)
    v0 = u2f(r0) * scale + lo
    v1 = u2f(r1) * scale + lo
    v2 = u2f(r2) * scale + lo
    v3 = u2f(r3) * scale + lo
    if CLAMP:
        v0 = tl.minimum(v0, hi)
        v1 = tl.minimum(v1, hi)
        v2 = tl.minimum(v2, hi)
        v3 = tl.minimum(v3, hi)
    base = tl.program_id(0) * BLOCK * 4 + lane
    if EVEN:
        tl.store(o + base, v0)
        tl.store(o + base + BLOCK, v1)
        tl.store(o + base + BLOCK * 2, v2)
        tl.store(o + base + BLOCK * 3, v3)
    else:
        tl.store(o + base, v0, mask=base < N)
        tl.store(o + base + BLOCK, v1, mask=base + BLOCK < N)
        tl.store(o + base + BLOCK * 2, v2, mask=base + BLOCK * 2 < N)
        tl.store(o + base + BLOCK * 3, v3, mask=base + BLOCK * 3 < N)


@triton.jit
def k_uni1(o, N, seed, poff, lo, hi, scale, BLOCK: tl.constexpr, EVEN: tl.constexpr,
           CLAMP: tl.constexpr):
    seed = seed.to(tl.int64)
    poff = poff.to(tl.int64)
    c0 = (poff & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((poff >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i
    _O = c0 * 0
    r0, _, _, _ = tl.philox(seed, c0, c1, _O, _O)
    v = u2f(r0) * scale + lo
    if CLAMP:
        v = tl.minimum(v, hi)
    if EVEN:
        tl.store(o + i, v)
    else:
        tl.store(o + i, v, mask=i < N)


def make(n, dtype):
    return torch.empty(n, dtype=dtype, device="npu")


def run_named(name, fn):
    try:
        res["variants"][name] = round(bench(fn), 3)
    except Exception:
        res["variants"][name] = "ERR:" + traceback.format_exc().splitlines()[-1][:200]


def main():
    dev = torch.npu.current_device()
    res["device"] = torch.npu.get_device_name(dev)
    try:
        p = torch.npu.get_device_properties(dev)
        res["props"] = {k: getattr(p, k, None) for k in dir(p) if not k.startswith("_")}
    except Exception:
        res["notes"].append("props: " + traceback.format_exc()[-200:])

    import flag_gems
    from flag_gems.runtime import torch_device_fn
    from flag_gems.utils.random_utils import philox_backend_seed_offset

    res["vendor"] = flag_gems.vendor_name
    x1024 = make(1024, torch.float16)
    gen = torch_device_fn.default_generators[dev]
    res["gen_type"] = type(gen).__name__

    run_named("host_empty_like_1024", lambda: torch.empty_like(x1024))

    def h_state():
        s = gen.get_state()
        c0, c1 = s.view(torch.int64)
        return int(c0), int(c1)

    run_named("host_gen_get_state", h_state)

    def h_state_set():
        s = gen.get_state()
        c0, c1 = s.view(torch.int64)
        c1 += 1024
        gen.set_state(s)

    run_named("host_gen_get_set_state", h_state_set)
    run_named("host_philox_seed_offset", lambda: philox_backend_seed_offset(1024))

    def h_guard():
        with torch_device_fn.device(x1024.device):
            pass

    run_named("host_device_guard", h_guard)

    seed_mod = None
    for cand in ("tmp/main.py", "main.py"):
        if os.path.exists(cand):
            try:
                spec = importlib.util.spec_from_file_location("seed_candidate", cand)
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
                seed_mod = m
                res["seed_module"] = cand
                break
            except Exception:
                res["notes"].append("seed import failed: " + traceback.format_exc()[-400:])
    if seed_mod is not None:
        for n, tag in ((1024, "f16"), (1048576, "f16")):
            xx = make(n, torch.float16)
            run_named(f"seed_full_{tag}_{n}", lambda xx=xx: seed_mod.uniform(xx, 0.0, 1.0))
            ps, po = philox_backend_seed_offset(triton.cdiv(n, 4))
            grid = (triton.cdiv(n, 1024 * 4),)

            def kern(xx=xx, ps=ps, po=po, n=n, grid=grid):
                seed_mod.uniform_kernel[grid](xx, n, ps, po, 0.0, 1.0, BLOCK=1024)

            try:
                kern()
                torch.npu.synchronize()
                run_named(f"seed_kernel_only_{tag}_{n}", kern)
            except Exception:
                res["variants"][f"seed_kernel_only_{tag}_{n}"] = "ERR:" + \
                    traceback.format_exc().splitlines()[-1][:160]

    # ---- candidate designs ----
    small_big = make(1048576, torch.float16)
    small_f32 = make(1048576, torch.float32)
    tiny = make(1024, torch.float16)
    tiny32 = make(1024, torch.float32)
    ps, po = philox_backend_seed_offset(4194304)

    for BLK in (1024, 4096, 8192, 16384):
        grid = (triton.cdiv(1048576, BLK * 4),)
        key = f"uni4_f16_1M_b{BLK}"
        try:
            k_uni4[grid](small_big, 1048576, ps, po, 0.0, 1.0, 1.0, BLOCK=BLK, EVEN=True,
                         CLAMP=True)
            torch.npu.synchronize()
            run_named(key, lambda grid=grid, BLK=BLK: k_uni4[grid](
                small_big, 1048576, ps, po, 0.0, 1.0, 1.0, BLOCK=BLK, EVEN=True, CLAMP=True))
        except Exception:
            res["variants"][key] = "ERR:" + traceback.format_exc().splitlines()[-1][:160]

    for BLK in (4096, 8192):
        grid = (triton.cdiv(1048576, BLK * 4),)
        key = f"uni4_f32_1M_b{BLK}"
        try:
            k_uni4[grid](small_f32, 1048576, ps, po, 0.0, 1.0, 1.0, BLOCK=BLK, EVEN=True,
                         CLAMP=True)
            torch.npu.synchronize()
            run_named(key, lambda grid=grid, BLK=BLK: k_uni4[grid](
                small_f32, 1048576, ps, po, 0.0, 1.0, 1.0, BLOCK=BLK, EVEN=True, CLAMP=True))
        except Exception:
            res["variants"][key] = "ERR:" + traceback.format_exc().splitlines()[-1][:160]

    for BLK in (1024, 4096):
        key = f"uni4_f16_1024_b{BLK}"
        try:
            k_uni4[(1,)](tiny, 1024, ps, po, 0.0, 1.0, 1.0, BLOCK=BLK, EVEN=False, CLAMP=True)
            torch.npu.synchronize()
            run_named(key, lambda BLK=BLK: k_uni4[(1,)](
                tiny, 1024, ps, po, 0.0, 1.0, 1.0, BLOCK=BLK, EVEN=False, CLAMP=True))
        except Exception:
            res["variants"][key] = "ERR:" + traceback.format_exc().splitlines()[-1][:160]

    key = "uni4_f32_1024_b1024"
    try:
        k_uni4[(1,)](tiny32, 1024, ps, po, 0.0, 1.0, 1.0, BLOCK=1024, EVEN=False, CLAMP=True)
        torch.npu.synchronize()
        run_named(key, lambda: k_uni4[(1,)](tiny32, 1024, ps, po, 0.0, 1.0, 1.0,
                                            BLOCK=1024, EVEN=False, CLAMP=True))
    except Exception:
        res["variants"][key] = "ERR:" + traceback.format_exc().splitlines()[-1][:160]

    # unroll=1 alternative and store-only / hash bounds
    for BLK in (8192,):
        grid = (triton.cdiv(1048576, BLK),)
        key = f"uni1_f16_1M_b{BLK}"
        try:
            k_uni1[grid](small_big, 1048576, ps, po, 0.0, 1.0, 1.0, BLOCK=BLK, EVEN=True,
                         CLAMP=True)
            torch.npu.synchronize()
            run_named(key, lambda grid=grid, BLK=BLK: k_uni1[grid](
                small_big, 1048576, ps, po, 0.0, 1.0, 1.0, BLOCK=BLK, EVEN=True, CLAMP=True))
        except Exception:
            res["variants"][key] = "ERR:" + traceback.format_exc().splitlines()[-1][:160]

    run_named("fill_f16_1M_b8192",
              lambda: k_fill[(triton.cdiv(1048576, 8192),)](small_big, 1048576, 0.5,
                                                           BLOCK=8192, EVEN=True))
    run_named("hash_f16_1M_b8192",
              lambda: k_hash[(triton.cdiv(1048576, 8192),)](small_big, 1048576, ps, po,
                                                           0.0, 1.0, BLOCK=8192, EVEN=True))
    run_named("fill_f16_1024",
              lambda: k_fill[(1,)](tiny, 1024, 0.5, BLOCK=8192, EVEN=False))
    run_named("empty_like_plus_fill_1024",
              lambda: (torch.empty_like(tiny),
                       k_fill[(1,)](tiny, 1024, 0.5, BLOCK=8192, EVEN=False)))

    with open(os.path.join(ART, "probe.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


main()
