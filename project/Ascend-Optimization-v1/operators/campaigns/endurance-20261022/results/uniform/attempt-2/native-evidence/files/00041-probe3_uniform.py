"""Probe 3: validate the new candidate (manual Philox + dtype clamp) and time it.

Sections: semantic mini-gate (mirrors the independent checker's uniform cases),
bit-equality against tl.philox, kernel/wrapper timings, BLOCK sweep, launch-path
introspection.
"""
import json
import os
import pathlib

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from flag_gems.utils.random_utils import uint_to_uniform_float

ART = pathlib.Path(os.environ.get("KGS_DEBUG_ARTIFACTS", "."))
ART.mkdir(parents=True, exist_ok=True)
R = {}


def load_candidate():
    import importlib.util

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
    seed, off = 987654321, 4096
    k_raw[grid](a, N, seed, off, BLOCK=BLOCK)
    k_raw_tl[grid](b, N, seed, off, BLOCK=BLOCK)
    torch.npu.synchronize()
    return {
        "equal": bool(torch.equal(a, b)),
        "mismatches": int((a != b).sum()),
        "sample_manual": [int(v) for v in a[:4].tolist()],
        "sample_tl": [int(v) for v in b[:4].tolist()],
    }


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


def timed(name, fn, iters=50):
    try:
        R[name] = round(bench_us(fn, iters), 2)
    except Exception as exc:
        R[name] = "ERROR: " + type(exc).__name__ + ": " + str(exc)[:400]


# ------------------------------------------------------------------ semantics
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
                    assert actual.is_contiguous(), "output not contiguous"
                    v = actual.detach().cpu().to(torch.float64)
                    assert torch.isfinite(v).all(), "non-finite"
                    if low == high:
                        assert bool((actual == low).all()), "constant range mismatch"
                    else:
                        assert bool((v >= low).all()) and bool((v < high).all()), (
                            f"range [{v.min().item()}, {v.max().item()}] outside [{low}, {high})")
                        z = (v - low) / (high - low)
                        assert abs(z.mean().item() - 0.5) < 0.025, "mean"
                        assert abs(z.var(unbiased=False).item() - 1 / 12) < 0.012, "var"
                        nxt = mod.run(x, low, high)
                        torch.npu.synchronize()
                        assert not torch.equal(nxt, actual), "reused output"
                row.update(status="passed", reps=2, stream_switch=True)
            except Exception as exc:
                row["error"] = type(exc).__name__ + ": " + str(exc)[:200]
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
                    offset = gen.get_offset()
                    assert offset > 0, "no state advance"
                    second = mod.run(probe, 0.0, 1.0)
                    torch.npu.synchronize()
                    assert not torch.equal(first, second), "consecutive calls repeat"
                    gen.set_state(state)
                    replay = mod.run(probe, 0.0, 1.0)
                    torch.npu.synchronize()
                    assert torch.equal(second, replay), "replay mismatch"
                row["status"] = "passed"
            except Exception as exc:
                row["error"] = type(exc).__name__ + ": " + str(exc)[:200]
            rows.append(row)
    finally:
        gen.set_state(saved)
    return rows


# ------------------------------------------------------------------- timings
def timings():
    for dtype, n, iters in [(torch.float32, 1024, 100), (torch.float32, 65536, 50),
                            (torch.float32, 1048576, 30), (torch.float16, 1048576, 30),
                            (torch.bfloat16, 1048576, 30)]:
        x = torch.empty(n, dtype=dtype, device="npu:0")
        timed(f"wrapper_{str(dtype).split('.')[-1]}_{n}",
              lambda x=x: mod.run(x, 0.0, 1.0), iters)

    # kernel-only launch cost, BLOCK sweep (no allocation, no state fetch)
    for dtype, n, blocks in [(torch.float32, 1024, [64, 256]), (torch.float32, 65536, [256, 512, 1024]),
                             (torch.float32, 1048576, [256, 512, 1024]),
                             (torch.float16, 1048576, [512, 1024])]:
        out = torch.empty(n, dtype=dtype, device="npu:0")
        for B in blocks:
            grid = triton.cdiv(n, B * 4)

            def launch(out=out, n=n, B=B, grid=grid):
                mod.uniform_kernel[(grid,)](out, n, 1234, 0, 0.0, 1.0, BLOCK=B)

            timed(f"kernel_{str(dtype).split('.')[-1]}_{n}_B{B}_g{grid}", launch, 30)


def launch_paths():
    out = torch.empty(4096, dtype=torch.float32, device="npu:0")
    grid = (1,)
    ret = mod.uniform_kernel[grid](out, 4096, 1234, 0, 0.0, 1.0, BLOCK=1024)
    info = {"type": str(type(ret)), "attrs": [a for a in dir(ret) if not a.startswith("_")][:40]}
    try:
        R["compiled_run_sig"] = str(__import__("inspect").signature(ret.run))
    except Exception as exc:
        R["compiled_run_sig"] = "ERROR: " + str(exc)[:200]
    try:
        ret[grid](out, 4096, 1234, 0, 0.0, 1.0)
        info["runner_getitem"] = "ok-no-BLOCK"
    except Exception as exc:
        info["runner_getitem"] = "ERROR: " + type(exc).__name__ + ": " + str(exc)[:200]
    R["compiled"] = info
    driver = triton.runtime.driver.active
    R["driver"] = {"type": type(driver).__name__,
                   "attrs": [a for a in dir(driver) if "stream" in a.lower() or "device" in a.lower()]}


for key, fn in [("bitcheck", bitcheck), ("semantics", semantics),
                ("generator", generator_semantics), ("timings", timings),
                ("launch", launch_paths)]:
    try:
        fn()
    except Exception as exc:
        import traceback
        R[key + "_ERROR"] = type(exc).__name__ + ": " + str(exc)[:600] + "\n" + traceback.format_exc()[-800:]

(ART / "probe3-uniform.json").write_text(json.dumps(R, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(R, indent=1, ensure_ascii=False)[:9000])
