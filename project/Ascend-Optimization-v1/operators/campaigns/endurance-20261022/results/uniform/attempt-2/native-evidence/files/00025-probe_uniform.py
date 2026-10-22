"""Bounded target diagnostic for uniform: host RNG-state cost and Philox device lowering.

Writes machine-readable results to $KGS_DEBUG_ARTIFACTS/probe-uniform.json.
"""
import hashlib
import inspect
import json
import os
import pathlib
import traceback

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = pathlib.Path(os.environ.get("KGS_DEBUG_ARTIFACTS", "."))
ART.mkdir(parents=True, exist_ok=True)
R = {}


def note(key, fn):
    try:
        R[key] = fn()
    except Exception as exc:  # keep going, record the failure
        R[key] = "ERROR: " + type(exc).__name__ + ": " + str(exc)


# ---------------------------------------------------------------- environment
note("torch", lambda: torch.__version__)
note("torch_npu", lambda: torch_npu.__version__)
note("triton", lambda: triton.__version__)
import flag_gems  # noqa: E402

note("flag_gems_file", lambda: flag_gems.__file__)
from flag_gems.utils import random_utils  # noqa: E402
from flag_gems.utils.random_utils import (  # noqa: E402
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

note("random_utils_file", lambda: random_utils.__file__)
note(
    "random_utils_sha256",
    lambda: hashlib.sha256(pathlib.Path(random_utils.__file__).read_bytes()).hexdigest(),
)
note("src_philox_backend_seed_offset", lambda: inspect.getsource(philox_backend_seed_offset))
note("src_uint_to_uniform_float", lambda: inspect.getsource(uint_to_uniform_float))

pkg = pathlib.Path(flag_gems.__file__).parent


def lean_source():
    for path in pkg.rglob("rrelu_with_noise.py"):
        text = path.read_text(encoding="utf-8")
        if "_lean_seed_offset" in text:
            lines = text.splitlines()
            start = next(i for i, line in enumerate(lines) if line.startswith("def _lean_seed_offset"))
            body = []
            for line in lines[start:]:
                if body and line and not line[0].isspace() and not line.startswith(")"):
                    break
                body.append(line)
            return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "lean_helper": "\n".join(body)}
    return "not found"


note("lean_seed_offset", lean_source)
note("heuristic_config", lambda: str(flag_gems.runtime.get_heuristic_config("uniform")))
note("device_props", lambda: str(torch.npu.get_device_properties(torch.npu.current_device())))


def vector_core_count():
    from triton.backends import backends

    import triton.runtime.driver as drv

    props = drv.active.get_current_device_properties() if hasattr(drv.active, "get_current_device_properties") else None
    return {"active": type(drv.active).__name__, "props": str(props), "backends": list(backends.keys())}


note("driver", vector_core_count)

# ------------------------------------------------------------- host RNG paths
INC = 16
gen = torch.npu.default_generators[torch.npu.current_device()]


def time_host(fn, iters=200):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    import time

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def parent_path():
    return philox_backend_seed_offset(INC, generator=None)


def lean_path():
    seed = gen.initial_seed()
    off = gen.get_offset()
    gen.set_offset(off + INC * 4)
    return seed, off


note("host_us_parent_seed_offset", lambda: time_host(parent_path))
note("host_us_lean_seed_offset", lambda: time_host(lean_path))
note("host_us_empty_like", lambda: time_host(lambda: torch.empty(4096, dtype=torch.float32, device="npu:0")))


def offset_delta_parent():
    gen.manual_seed(1234)
    gen.set_offset(0)
    before = gen.get_offset()
    philox_backend_seed_offset(INC, generator=None)
    return gen.get_offset() - before


def offset_delta_lean():
    gen.manual_seed(1234)
    gen.set_offset(0)
    before = gen.get_offset()
    lean_path()
    return gen.get_offset() - before


note("offset_delta_parent", offset_delta_parent)
note("offset_delta_lean", offset_delta_lean)
note("seed_and_offset_from_parent", lambda: [int(v) for v in parent_path()])
note("get_state_us", lambda: time_host(lambda: gen.get_state()))
note("set_state_us", lambda: time_host(lambda: gen.set_state(gen.get_state())))

# ------------------------------------------------------------ kernel variants
M0 = 0xD2511F53
M1 = 0xCD9E8D57
W0 = 0x9E3779B9
W1 = 0xBB67AE85


@triton.jit
def _mulhi16(a, B0: tl.constexpr, B1: tl.constexpr):
    a0 = a & 0xFFFF
    a1 = a >> 16
    w0 = a0 * B0
    t = a1 * B0 + (w0 >> 16)
    w1 = a0 * B1 + (t & 0xFFFF)
    return a1 * B1 + (t >> 16) + (w1 >> 16)


@triton.jit
def _philox16(seed, c0, c1):
    k0 = seed.to(tl.uint32)  # low 32 bits of the 64-bit seed
    k1 = (seed >> 32).to(tl.uint32)
    c2 = c0 * 0
    c3 = c2
    for _ in tl.static_range(10):
        hi0 = _mulhi16(c0, M0 & 0xFFFF, M0 >> 16)
        lo0 = c0 * (M0 & 0xFFFFFFFF)
        hi1 = _mulhi16(c2, M1 & 0xFFFF, M1 >> 16)
        lo1 = c2 * (M1 & 0xFFFFFFFF)
        c0 = hi1 ^ c1 ^ k0
        c1 = lo1
        c2 = hi0 ^ c3 ^ k1
        c3 = lo0
        k0 = k0 + (W0 & 0xFFFFFFFF)
        k1 = k1 + (W1 & 0xFFFFFFFF)
    return c0, c1, c2, c3


@triton.jit
def _philox64(seed, c0, c1):
    k0 = seed.to(tl.uint32)
    k1 = (seed >> 32).to(tl.uint32)
    c2 = c0 * 0
    c3 = c2
    for _ in tl.static_range(10):
        p0 = c0.to(tl.uint64) * M0
        p1 = c2.to(tl.uint64) * M1
        hi0 = (p0 >> 32).to(tl.uint32)
        lo0 = p0.to(tl.uint32)
        hi1 = (p1 >> 32).to(tl.uint32)
        lo1 = p1.to(tl.uint32)
        c0 = hi1 ^ c1 ^ k0
        c1 = lo1
        c2 = hi0 ^ c3 ^ k1
        c3 = lo0
        k0 = k0 + (W0 & 0xFFFFFFFF)
        k1 = k1 + (W1 & 0xFFFFFFFF)
    return c0, c1, c2, c3


@triton.jit
def _next_below(x, DT: tl.constexpr):
    zero = x == 0
    if DT == tl.float32:
        bits = x.to(tl.int32, bitcast=True)
        step = tl.where(x >= 0, -1, 1).to(tl.int32)
        stepped = (bits + step).to(tl.float32, bitcast=True)
        tiny = tl.full((), 1, tl.int32).to(tl.float32, bitcast=True)
        return tl.where(zero, -tiny, stepped).to(tl.float32)
    else:
        bits = x.to(tl.int16, bitcast=True)
        step = tl.where(x >= 0, -1, 1).to(tl.int16)
        stepped = (bits + step).to(DT, bitcast=True)
        tiny = tl.full((), 1, tl.int16).to(DT, bitcast=True)
        return tl.where(zero, -tiny, stepped).to(DT)


@triton.jit
def _emit(r, out_ptr, off, N, from_, to):
    OUT_DT: tl.constexpr = out_ptr.dtype.element_ty
    to_dt = (r * 0.0 + to).to(OUT_DT)
    from_dt = (r * 0.0 + from_).to(OUT_DT)
    if OUT_DT == tl.float32:
        hi = _next_below(to_dt, OUT_DT)
    else:
        hi = _next_below(to_dt, OUT_DT)
    hi = tl.where(to_dt > from_dt, hi, to_dt)
    v = tl.minimum(r.to(OUT_DT), hi)
    tl.store(out_ptr + off, v, mask=off < N)


@triton.jit
def k_const(out_ptr, N, from_, to, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK * 4 + tl.arange(0, BLOCK)
    v = (from_ * 0.0 + 0.5).to(out_ptr.dtype.element_ty) + tl.zeros((BLOCK,), tl.float32).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + off, v, mask=off < N)
    tl.store(out_ptr + off + BLOCK, v, mask=off + BLOCK < N)
    tl.store(out_ptr + off + BLOCK * 2, v, mask=off + BLOCK * 2 < N)
    tl.store(out_ptr + off + BLOCK * 3, v, mask=off + BLOCK * 3 < N)


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def k_tlphilox(out_ptr, N, philox_seed, philox_offset, from_, to, BLOCK: tl.constexpr):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    off = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    _emit(uint_to_uniform_float(r0) * (to - from_) + from_, out_ptr, off, N, from_, to)
    _emit(uint_to_uniform_float(r1) * (to - from_) + from_, out_ptr, off + BLOCK, N, from_, to)
    _emit(uint_to_uniform_float(r2) * (to - from_) + from_, out_ptr, off + BLOCK * 2, N, from_, to)
    _emit(uint_to_uniform_float(r3) * (to - from_) + from_, out_ptr, off + BLOCK * 3, N, from_, to)


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def k_manual16(out_ptr, N, philox_seed, philox_offset, from_, to, BLOCK: tl.constexpr):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    r0, r1, r2, r3 = _philox16(philox_seed, c0, c1)
    off = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    _emit(uint_to_uniform_float(r0) * (to - from_) + from_, out_ptr, off, N, from_, to)
    _emit(uint_to_uniform_float(r1) * (to - from_) + from_, out_ptr, off + BLOCK, N, from_, to)
    _emit(uint_to_uniform_float(r2) * (to - from_) + from_, out_ptr, off + BLOCK * 2, N, from_, to)
    _emit(uint_to_uniform_float(r3) * (to - from_) + from_, out_ptr, off + BLOCK * 3, N, from_, to)


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def k_manual64(out_ptr, N, philox_seed, philox_offset, from_, to, BLOCK: tl.constexpr):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    r0, r1, r2, r3 = _philox64(philox_seed, c0, c1)
    off = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    _emit(uint_to_uniform_float(r0) * (to - from_) + from_, out_ptr, off, N, from_, to)
    _emit(uint_to_uniform_float(r1) * (to - from_) + from_, out_ptr, off + BLOCK, N, from_, to)
    _emit(uint_to_uniform_float(r2) * (to - from_) + from_, out_ptr, off + BLOCK * 2, N, from_, to)
    _emit(uint_to_uniform_float(r3) * (to - from_) + from_, out_ptr, off + BLOCK * 3, N, from_, to)


BLOCK = 1024


def bench_us(fn, iters):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    e0 = torch.npu.Event(enable_timing=True)
    e1 = torch.npu.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.npu.synchronize()
    return e0.elapsed_time(e1) * 1e3 / iters


def run_variant(kernel, N, dtype, iters):
    out = torch.empty(N, dtype=dtype, device="npu:0")
    grid = (triton.cdiv(N, BLOCK * 4),)
    seed, off = 1234, 0
    fn = lambda: kernel[grid](out, N, seed, off, 0.0, 1.0, BLOCK=BLOCK)
    us = bench_us(fn, iters)
    return {"us_per_call": us, "grid": grid[0], "n": N, "dtype": str(dtype)}


ALL = [("const", k_const), ("tl_philox", k_tlphilox),
       ("manual16", k_manual16), ("manual64", k_manual64)]
for dtype, N, iters, variants in [
    (torch.float32, 1048576, 20, ALL),
    (torch.float32, 1024, 200, ALL[:2]),
    (torch.float16, 1048576, 20, ALL[1:3]),
]:
    tag = f"{str(dtype).split('.')[-1]}_{N}"
    for name, kern in variants:
        note(f"kern_{tag}_{name}", lambda kern=kern, N=N, dtype=dtype, iters=iters: run_variant(kern, N, dtype, iters))


# ------------------------------------------------- bit equality with tl.philox
def bitcheck():
    N = 4096
    out_a = torch.empty(N, dtype=torch.float32, device="npu:0")
    out_b = torch.empty(N, dtype=torch.float32, device="npu:0")
    out_c = torch.empty(N, dtype=torch.float32, device="npu:0")
    grid = (triton.cdiv(N, BLOCK * 4),)
    seed, off = 987654321, 4096
    k_tlphilox[grid](out_a, N, seed, off, 0.0, 1.0, BLOCK=BLOCK)
    k_manual16[grid](out_b, N, seed, off, 0.0, 1.0, BLOCK=BLOCK)
    k_manual64[grid](out_c, N, seed, off, 0.0, 1.0, BLOCK=BLOCK)
    torch.npu.synchronize()
    return {
        "manual16_equal": bool(torch.equal(out_a, out_b)),
        "manual64_equal": bool(torch.equal(out_a, out_c)),
        "manual16_maxdiff": float((out_a - out_b).abs().max()),
        "manual64_maxdiff": float((out_a - out_c).abs().max()),
        "range": [float(out_a.min()), float(out_a.max())],
        "mean_a": float(out_a.double().mean()),
    }


note("bitcheck", bitcheck)


# --------------------------------------------- full wrapper walltime (parent)
def wrapper_times():
    import importlib.util

    cand = pathlib.Path("tmp/main.py")
    if not cand.is_file():
        cand = next(pathlib.Path.cwd().rglob("tmp/main.py"))
    spec = importlib.util.spec_from_file_location("cand", str(cand))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    res = {}
    for dtype, N in [(torch.float32, 1024), (torch.float32, 1048576)]:
        x = torch.empty(N, dtype=dtype, device="npu:0")
        res[f"parent_wrapper_{N}"] = bench_us(lambda x=x: mod.run(x, 0.0, 1.0), 30 if N > 4096 else 100)
    return res


note("wrapper", wrapper_times)

(ART / "probe-uniform.json").write_text(json.dumps(R, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({k: (v if not isinstance(v, str) else v[:120]) for k, v in R.items()}, ensure_ascii=False, indent=1)[:6000])
