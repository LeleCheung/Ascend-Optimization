"""Probe 5: validate candidate v2 (host-side dtype bounds), verify the manual
Philox against a Random123 reference, and discover/timing a direct
CompiledKernel launch path."""
import importlib.util
import json
import os
import pathlib
import struct
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = pathlib.Path(os.environ.get("KGS_DEBUG_ARTIFACTS", "."))
ART.mkdir(parents=True, exist_ok=True)
R = {}


def load_candidate():
    path = pathlib.Path("tmp/main.py")
    if not path.is_file():
        path = next(pathlib.Path.cwd().rglob("tmp/main.py"))
    spec = importlib.util.spec_from_file_location("cand", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = load_candidate()

# ------------------------------------------------------- Random123 reference
M0, M1 = 0xD2511F53, 0xCD9E8D57
W0, W1 = 0x9E3779B9, 0xBB67AE85


def py_philox(c0, c1, c2, c3, k0, k1, rounds=10):
    for _ in range(rounds):
        p0 = (M0 * c0) & 0xFFFFFFFFFFFFFFFF
        p1 = (M1 * c2) & 0xFFFFFFFFFFFFFFFF
        hi0, lo0 = (p0 >> 32) & 0xFFFFFFFF, p0 & 0xFFFFFFFF
        hi1, lo1 = (p1 >> 32) & 0xFFFFFFFF, p1 & 0xFFFFFFFF
        c0 = (hi1 ^ c1 ^ k0) & 0xFFFFFFFF
        c1 = lo1
        c2 = (hi0 ^ c3 ^ k1) & 0xFFFFFFFF
        c3 = lo0
        k0 = (k0 + W0) & 0xFFFFFFFF
        k1 = (k1 + W1) & 0xFFFFFFFF
    return c0, c1, c2, c3


def mulhi16_identity():
    import random
    rng = random.Random(7)
    bad = 0
    for _ in range(100000):
        a = rng.getrandbits(32)
        for B in (M0, M1):
            a0, a1 = a & 0xFFFF, a >> 16
            B0, B1 = B & 0xFFFF, B >> 16
            u0 = a0 * B0
            t = a1 * B0 + (u0 >> 16)
            w1 = a0 * B1 + (t & 0xFFFF)
            hi = (a1 * B1 + (t >> 16) + (w1 >> 16)) & 0xFFFFFFFF
            lo = (u0 ^ ((w1 << 16) & 0xFFFFFFFF)) & 0xFFFFFFFF
            p = a * B
            if hi != (p >> 32) or lo != (p & 0xFFFFFFFF):
                bad += 1
    return {"checked": 200000, "mismatches": bad}


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def k_raw(out_ptr, N, philox_seed, philox_offset, BLOCK: tl.constexpr):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r0, r1, r2, r3 = mod._philox4(philox_seed, c0 + lane, c1)
    off = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    tl.store(out_ptr + off, r0.to(tl.int32, bitcast=True), mask=off < N)
    tl.store(out_ptr + off + BLOCK, r1.to(tl.int32, bitcast=True), mask=off + BLOCK < N)
    tl.store(out_ptr + off + BLOCK * 2, r2.to(tl.int32, bitcast=True), mask=off + BLOCK * 2 < N)
    tl.store(out_ptr + off + BLOCK * 3, r3.to(tl.int32, bitcast=True), mask=off + BLOCK * 3 < N)


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def k_raw_tl(out_ptr, N, philox_seed, philox_offset, BLOCK: tl.constexpr):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 = c0 + lane
    o = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, o, o)
    off = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    tl.store(out_ptr + off, r0, mask=off < N)
    tl.store(out_ptr + off + BLOCK, r1, mask=off + BLOCK < N)
    tl.store(out_ptr + off + BLOCK * 2, r2, mask=off + BLOCK * 2 < N)
    tl.store(out_ptr + off + BLOCK * 3, r3, mask=off + BLOCK * 3 < N)


def philox_reference():
    n, block = 64, 16
    out = torch.empty(n, dtype=torch.int32, device="npu:0")
    out_tl = torch.empty(n, dtype=torch.int32, device="npu:0")
    seed, offset = 0x123456789ABCDEF0, 0x0000000100000000
    grid = (triton.cdiv(n, block * 4),)
    k_raw[grid](out, n, seed, offset, BLOCK=block)
    k_raw_tl[grid](out_tl, n, seed, offset, BLOCK=block)
    torch.npu.synchronize()
    got = [int(v) & 0xFFFFFFFF for v in out.tolist()]
    got_tl = [int(v) & 0xFFFFFFFF for v in out_tl.tolist()]
    k0, k1 = seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF
    ref = []
    for lane in range(n // 4):
        c0 = ((offset & 0xFFFFFFFF) + lane) & 0xFFFFFFFF
        c1 = (offset >> 32) & 0xFFFFFFFF
        ref.extend(py_philox(c0, c1, 0, 0, k0, k1))
    return {
        "manual_matches_random123": got == ref,
        "tl_matches_random123": got_tl == ref,
        "manual_first4": got[:4],
        "tl_first4": got_tl[:4],
        "ref_first4": ref[:4],
    }


def bounds_matrix():
    lo_hi = [(0.0, 1.0), (-3.0, 2.0), (0.5, 0.5), (-0.01, 0.02), (-100.0, -10.0),
             (0.1, 0.9), (1e-30, 1e-29), (-0.0, 1.0), (0.4999999, 0.5000001)]
    rows = {}
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        for lo, hi in lo_hi:
            a, b = mod._sample_bounds(dtype, lo, hi)
            ok_exact = (mod._round_dtype(a, dtype) == a) and (mod._round_dtype(b, dtype) == b)
            rows[f"{str(dtype).split('.')[-1]}_{lo}_{hi}"] = {
                "lo": a, "hi": b, "exact": ok_exact,
                "lo_ge_from": (a >= lo) if lo != hi else (a == lo or a == mod._round_dtype(lo, dtype)),
                "hi_lt_to": (b < hi) if lo != hi else True,
                "ordered": a <= b,
            }
    return rows


def semantics():
    rows = []
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        for case in range(6):
            row = dict(dtype=str(dtype), case=case, status="failed")
            try:
                g = torch.Generator().manual_seed(73137 + case)
                shape = (65537,) if case < 4 else (127, 513)
                x = torch.randn(shape, generator=g).to(device="npu:0", dtype=dtype)
                if case == 5:
                    base = torch.full((*x.shape, 2), 17.0, device=x.device, dtype=x.dtype)
                    base[..., 0] = x
                    x = base[..., 0]
                low, high = [(0.0, 1.0), (-3.0, 2.0), (0.5, 0.5), (-0.01, 0.02),
                             (-100.0, -10.0), (0.1, 0.9)][case]
                before = x.clone()
                stream = torch.npu.Stream()
                for rep in range(2):
                    torch.npu.synchronize()
                    with torch.npu.stream(stream if rep else torch.npu.default_stream()):
                        actual = mod.run(x, low, high)
                    torch.npu.synchronize()
                    assert not torch._C._is_alias_of(actual, x), "aliases input"
                    assert torch.equal(x, before), "input mutated"
                    assert actual.shape == x.shape and actual.dtype == dtype
                    assert actual.is_contiguous(), "not contiguous"
                    v = actual.detach().cpu().to(torch.float64)
                    assert torch.isfinite(v).all(), "non-finite"
                    if low == high:
                        assert bool((actual == low).all()), "constant mismatch"
                    else:
                        assert bool((v >= low).all()) and bool((v < high).all()), (
                            f"range [{v.min().item()}, {v.max().item()}] not in [{low}, {high})")
                        z = (v - low) / (high - low)
                        assert abs(z.mean().item() - 0.5) < 0.025, f"mean {z.mean().item()}"
                        assert abs(z.var(unbiased=False).item() - 1 / 12) < 0.012, "var"
                        nxt = mod.run(x, low, high)
                        torch.npu.synchronize()
                        assert not torch.equal(nxt, actual), "reused output"
                row.update(status="passed", reps=2, stream_switch=True)
            except Exception as exc:
                row["error"] = type(exc).__name__ + ": " + str(exc)[:220]
            rows.append(row)
    return rows


def generator_semantics():
    gen = torch.npu.default_generators[torch.npu.current_device()]
    saved = gen.get_state()
    rows = []
    try:
        probe = torch.empty(4096, dtype=torch.float32, device="npu:0")

        def sample(seed, offset):
            gen.manual_seed(seed)
            gen.set_offset(offset)
            res = mod.run(probe, 0.0, 1.0)
            torch.npu.synchronize()
            return res

        for name in ["seed_high_bits", "offset_high_bits", "state_advance_and_replay"]:
            row = dict(case=name, status="failed")
            try:
                if name == "seed_high_bits":
                    first = sample(12345, 0)
                    assert torch.equal(first, sample(12345, 0)), "not reproducible"
                    for bit in [31, 32, 40]:
                        assert not torch.equal(first, sample(12345 + (1 << bit), 0)), f"seed bit{bit}"
                elif name == "offset_high_bits":
                    first = sample(12345, 0)
                    for bit in [31, 32, 40]:
                        assert not torch.equal(first, sample(12345, 1 << bit)), f"offset bit{bit}"
                else:
                    first = sample(12345, 0)
                    state = gen.get_state()
                    assert gen.get_offset() > 0, "no advance"
                    second = mod.run(probe, 0.0, 1.0)
                    torch.npu.synchronize()
                    assert not torch.equal(first, second), "consecutive repeat"
                    gen.set_state(state)
                    replay = mod.run(probe, 0.0, 1.0)
                    torch.npu.synchronize()
                    assert torch.equal(second, replay), "replay mismatch"
                row["status"] = "passed"
            except Exception as exc:
                row["error"] = type(exc).__name__ + ": " + str(exc)[:220]
            rows.append(row)
    finally:
        gen.set_state(saved)
    return rows


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


def cpu_us(fn, iters=300):
    for _ in range(5):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1e6 / iters


def fast_launcher():
    """Discover a direct CompiledKernel launch form and time it."""
    out = {}
    n, block = 65536, 512
    grid = triton.cdiv(n, block * 4)
    x = torch.empty(n, dtype=torch.float32, device="npu:0")
    lo, hi = mod._sample_bounds(torch.float32, 0.0, 1.0)
    span = 1.0
    seed, offset = 12345, 0
    kern = mod.uniform_kernel[(grid,)](x, n, seed, offset, 0.0, span, lo, hi, BLOCK=block)
    torch.npu.synchronize()
    out["attrs"] = [a for a in dir(kern) if not a.startswith("_")]
    try:
        drv = triton.runtime.driver.active
        stream = drv.get_current_stream(torch.npu.current_device())
        out["stream_repr"] = f"{type(stream)}:{stream}"
    except Exception as exc:
        out["stream_error"] = type(exc).__name__ + ": " + str(exc)[:200]
        return out

    args = (x, n, seed, offset, 0.0, span, lo, hi)
    variants = {
        "hooks_none": lambda: kern.run(grid, 1, 1, stream, kern.function, kern.packed_metadata,
                                       None, None, None, *args),
        "meta_noargs_hooks": lambda: kern.run(grid, 1, 1, stream, kern.function, kern.packed_metadata,
                                              kern.launch_metadata, None, None, *args),
        "no_hooks": lambda: kern.run(grid, 1, 1, stream, kern.function, kern.packed_metadata, *args),
        "kernel_attr": lambda: kern.kernel.run(grid, 1, 1, stream, kern.function, kern.packed_metadata,
                                               None, None, None, *args),
    }
    for name, fn in variants.items():
        try:
            ref = mod.run(x.clone(), 0.0, 1.0)
            out[name] = {"status": "ran", "us": round(bench_us(fn, 50), 2),
                         "cpu_us": round(cpu_us(fn, 100), 2)}
            torch.npu.synchronize()
        except Exception as exc:
            out[name] = "ERROR: " + type(exc).__name__ + ": " + str(exc)[:300]

    # correctness of the working variant: same seed/offset as a normal launch
    try:
        a = torch.empty(n, dtype=torch.float32, device="npu:0")
        b = torch.empty(n, dtype=torch.float32, device="npu:0")
        mod.uniform_kernel[(grid,)](a, n, seed, offset, 0.0, span, lo, hi, BLOCK=block)
        kern.run(grid, 1, 1, stream, kern.function, kern.packed_metadata, None, None, None, *(
            b, n, seed, offset, 0.0, span, lo, hi))
        torch.npu.synchronize()
        out["equal_normal_vs_fast"] = bool(torch.equal(a, b))
    except Exception as exc:
        out["equal_normal_vs_fast"] = "ERROR: " + type(exc).__name__ + ": " + str(exc)[:200]

    # wrapper timings with the current candidate
    for dtype, nn, iters in [(torch.float32, 1024, 100), (torch.float32, 65536, 100),
                             (torch.float32, 1048576, 50), (torch.float16, 1048576, 50)]:
        t = torch.empty(nn, dtype=dtype, device="npu:0")
        out[f"wrapper_{str(dtype).split('.')[-1]}_{nn}_us"] = round(bench_us(lambda t=t: mod.run(t, 0.0, 1.0), iters), 2)
        out[f"wrapper_{str(dtype).split('.')[-1]}_{nn}_cpu_us"] = round(cpu_us(lambda t=t: mod.run(t, 0.0, 1.0), 100), 2)
    return out


for key, fn in [("mulhi16_identity", mulhi16_identity), ("philox_ref", philox_reference),
                ("bounds", bounds_matrix), ("semantics", semantics),
                ("generator", generator_semantics), ("launch", fast_launcher)]:
    try:
        R[key] = fn()
    except Exception as exc:
        import traceback
        R[key] = "ERROR: " + type(exc).__name__ + ": " + str(exc)[:400] + "\n" + traceback.format_exc()[-500:]

(ART / "probe5-uniform.json").write_text(json.dumps(R, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(R, indent=1, ensure_ascii=False)[:9000])
