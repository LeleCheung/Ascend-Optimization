"""Bounded target diagnostic: host/device split and arithmetic-domain timing."""
import importlib.util
import json
import os
import time

import torch
import torch_npu
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
RES = {}


def load_candidate():
    spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cand = load_candidate()
DEV = torch.npu.current_device()


@triton.jit
def _noop(P):
    tl.store(P + tl.program_id(0), 0)


@triton.jit
def _v2_flat(A, B, N: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = o < N
    a = tl.load(A + o, mask, other=0).to(tl.float32)
    b = tl.load(B + o, mask, other=1).to(tl.float32)
    q = a / b
    magnitude = tl.floor(tl.abs(q))
    truncated = tl.where(q < 0, -magnitude, magnitude)
    tl.store(A + o, a - truncated * b, mask)


@triton.jit
def _narrow_flat(A, B, N: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = o < N
    a = tl.load(A + o, mask, other=0)
    b = tl.load(B + o, mask, other=1)
    q = a / b
    magnitude = tl.floor(tl.abs(q))
    truncated = tl.where(q < 0, -magnitude, magnitude)
    tl.store(A + o, a - truncated * b, mask)


def make(n, dtype, seed=1729):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, generator=g).to(device=DEV, dtype=dtype)
    B = (torch.randn(n, generator=g).abs() + 0.05).to(device=DEV, dtype=dtype)
    return A, B


def time_loop(fn, iters):
    for _ in range(20):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def time_event(fn, iters):
    for _ in range(20):
        fn()
    torch.npu.synchronize()
    s = torch.npu.Event(enable_timing=True)
    e = torch.npu.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.npu.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


def bench(fn, iters):
    return {"loop_us": time_loop(fn, iters), "event_us": time_event(fn, iters)}


CASES = [(1024, torch.float16), (1024, torch.float32), (1048576, torch.bfloat16),
         (16777216, torch.float16), (16777216, torch.float32)]

for n, dt in CASES:
    tag = f"{n}_{str(dt).replace('torch.', '')}"
    iters = 300 if n <= 1048576 else 60
    case = {}
    A, B = make(n, dt)
    ref = bench(lambda: A.fmod_(B), iters)
    case["ref_fmod_"] = ref
    case["cand_run"] = bench(lambda: cand.run(A, B), iters)
    key = (A.dtype, B.dtype, A.device.index, n)
    entry = cand._ENTRIES.get(key)
    if entry is not None:
        launcher, fnt, pmd = entry
        grid = (n + cand._BLOCK - 1) // cand._BLOCK
        pa, pb = A.data_ptr(), B.data_ptr()
        case["prebound_launch"] = bench(
            lambda: launcher(grid, 1, 1, cand._stream(A.device.index), fnt, pmd,
                             None, None, None, pa, pb, n, cand._BLOCK), iters)
    else:
        case["prebound_launch"] = None
    buf = torch.empty(64, dtype=torch.float32, device=DEV)
    case["noop_launch"] = bench(lambda: _noop[(1,)](buf), iters)
    case["empty_like"] = bench(lambda: torch.empty_like(A), iters)
    RES[tag] = case

# micro: stream getter and misc python costs
torch.npu.synchronize()
t0 = time.perf_counter()
for _ in range(20000):
    cand._stream(A.device.index)
RES["stream_getter_ns"] = (time.perf_counter() - t0) / 20000 * 1e9
t0 = time.perf_counter()
for _ in range(20000):
    A.is_contiguous() and B.is_contiguous()
RES["contig_check_ns"] = (time.perf_counter() - t0) / 20000 * 1e9
t0 = time.perf_counter()
for _ in range(20000):
    A.numel()
RES["numel_ns"] = (time.perf_counter() - t0) / 20000 * 1e9
t0 = time.perf_counter()
for _ in range(20000):
    A.data_ptr()
RES["data_ptr_ns"] = (time.perf_counter() - t0) / 20000 * 1e9

# variant comparison: seed vs cheap-trunc fp32 vs narrow-domain arithmetic
VARIANTS = {"seed": cand._fmod_flat, "v2_trunc": _v2_flat, "narrow": _narrow_flat}
for n, dt in [(1048576, torch.bfloat16), (1048576, torch.float16),
              (16777216, torch.float16), (16777216, torch.float32),
              (16777216, torch.bfloat16)]:
    tag = f"var_{n}_{str(dt).replace('torch.', '')}"
    A0, B = make(n, dt)
    refout = None
    row = {}
    for name, kern in VARIANTS.items():
        A = A0.clone()
        try:
            grid = (n + 4096 - 1) // 4096
            fn = lambda: kern[(grid,)](A, B, n, 4096, enable_fp_fusion=False)
            fn()
            torch.npu.synchronize()
            cur = A.clone()
            if refout is None:
                R = A0.clone()
                R.fmod_(B)
                torch.npu.synchronize()
                refout = R
            diff = (cur.float() - refout.float()).abs()
            bad = int((diff > (1e-4 + 1e-4 * refout.float().abs())).sum().item())
            row[name] = {"loop_us": time_loop(fn, 60 if n > 1048576 else 300),
                         "event_us": time_event(fn, 60 if n > 1048576 else 300),
                         "max_abs_err": float(diff.max().item()),
                         "n_bad_1e-4": bad}
        except Exception as exc:  # compile/runtime failure is informative
            row[name] = {"error": f"{type(exc).__name__}: {str(exc)[:400]}"}
    RES[tag] = row

with open(os.path.join(ART, "probe.json"), "w") as fh:
    json.dump(RES, fh, indent=1)
print(json.dumps(RES, indent=1))
