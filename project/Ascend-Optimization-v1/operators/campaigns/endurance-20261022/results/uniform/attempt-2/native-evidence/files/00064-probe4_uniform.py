"""Probe 4: validate candidate semantics/RNG (properly reported), attribute host cost,
and sweep larger BLOCK values."""
import importlib.util
import json
import os
import pathlib
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import uint_to_uniform_float

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


def bitcheck():
    N, BLOCK = 4096, 1024
    a = torch.empty(N, dtype=torch.int32, device="npu:0")
    b = torch.empty(N, dtype=torch.int32, device="npu:0")
    grid = (triton.cdiv(N, BLOCK * 4),)
    k_raw[grid](a, N, 987654321, 4096, BLOCK=BLOCK)
    k_raw_tl[grid](b, N, 987654321, 4096, BLOCK=BLOCK)
    torch.npu.synchronize()
    # second: high bits of seed and offset must matter
    a2 = torch.empty(N, dtype=torch.int32, device="npu:0")
    k_raw[grid](a2, N, 987654321 + (1 << 40), 4096, BLOCK=BLOCK)
    a3 = torch.empty(N, dtype=torch.int32, device="npu:0")
    k_raw[grid](a3, N, 987654321, 4096 + (1 << 40), BLOCK=BLOCK)
    torch.npu.synchronize()
    return {
        "equal_tl_philox": bool(torch.equal(a, b)),
        "mismatches": int((a != b).sum()),
        "tl_matches_python_known": [int(v) for v in b[:4].tolist()],
        "manual_first4": [int(v) for v in a[:4].tolist()],
        "seed_high_bit_changes": bool(not torch.equal(a, a2)),
        "offset_high_bit_changes": bool(not torch.equal(a, a3)),
    }


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
                    assert actual.device == x.device
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


def host_attribution():
    x = torch.empty(4096, dtype=torch.float32, device="npu:0")
    gen = torch.npu.default_generators[torch.npu.current_device()]
    idx = torch.npu.current_device()
    out = {}
    out["empty_kwargs_cpu_us"] = round(cpu_us(lambda: torch.empty(x.shape, dtype=x.dtype, device=x.device)), 2)
    out["empty_like_cpu_us"] = round(cpu_us(lambda: torch.empty_like(x)), 2)
    out["gen_lookup_cpu_us"] = round(cpu_us(lambda: torch_device_fn.default_generators[idx]), 2)
    out["gen_state_cpu_us"] = round(cpu_us(lambda: (gen.initial_seed(), gen.get_offset(), gen.set_offset(gen.get_offset()))), 2)
    out["device_attr_cpu_us"] = round(cpu_us(lambda: (x.device, x.device.index)), 2)
    out["cdiv_cpu_us"] = round(cpu_us(lambda: min(1024, max(64, triton.next_power_of_2(triton.cdiv(4096, 4))))), 2)
    out["shape_numel_cpu_us"] = round(cpu_us(lambda: (x.shape, x.numel())), 2)
    # device-side walltime of the launch and of the whole wrapper
    out["launch_only_us"] = round(bench_us(lambda: mod.uniform_kernel[(1,)](x, 4096, 1234, 0, 0.0, 1.0, BLOCK=1024), 100), 2)
    out["wrapper_us"] = round(bench_us(lambda: mod.run(x, 0.0, 1.0), 100), 2)
    out["launch_cpu_us"] = round(cpu_us(lambda: mod.uniform_kernel[(1,)](x, 4096, 1234, 0, 0.0, 1.0, BLOCK=1024), 200), 2)
    out["wrapper_cpu_us"] = round(cpu_us(lambda: mod.run(x, 0.0, 1.0), 200), 2)
    return out


def block_sweep():
    rows = {}
    for dtype, n in [(torch.float32, 1048576), (torch.float16, 1048576), (torch.float32, 65536)]:
        out = torch.empty(n, dtype=dtype, device="npu:0")
        for B in [512, 1024, 2048, 4096]:
            if B * 4 > 4 * n:
                continue
            grid = triton.cdiv(n, B * 4)
            key = f"{str(dtype).split('.')[-1]}_{n}_B{B}_g{grid}"
            try:
                rows[key] = round(bench_us(
                    lambda out=out, n=n, B=B, grid=grid: mod.uniform_kernel[(grid,)](out, n, 1234, 0, 0.0, 1.0, BLOCK=B), 30), 2)
            except Exception as exc:
                rows[key] = "ERROR: " + type(exc).__name__ + ": " + str(exc)[:200]
    return rows


for key, fn in [("bitcheck", bitcheck), ("semantics", semantics),
                ("generator", generator_semantics), ("host", host_attribution),
                ("blocks", block_sweep)]:
    try:
        R[key] = fn()
    except Exception as exc:
        import traceback
        R[key] = "ERROR: " + type(exc).__name__ + ": " + str(exc)[:500] + "\n" + traceback.format_exc()[-600:]

(ART / "probe4-uniform.json").write_text(json.dumps(R, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(R, indent=1, ensure_ascii=False)[:9000])
