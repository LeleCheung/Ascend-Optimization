"""Bounded target-side probe for hardshrink_backward (<=120 s).

Measures, on the real device:
  1. per-call walltime of the ATen reference
  2. per-call walltime of the seed FlagGems/Triton run()
  3. host-only overhead pieces (empty_like, current_stream, raw stream, contiguous)
  4. alternative launch/blocking variants (direct compiled launcher, bigger tiles)
  5. compiled-launcher introspection (arg list / metadata) for the direct path
  6. numerical check of candidate formulas vs ATen on special-value inputs
"""
import inspect
import json
import os
import sys
import time
import traceback

import torch
import torch_npu
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
os.makedirs(ART, exist_ok=True)
OUT = {"cases": {}, "host": {}, "launcher": {}, "numerics": {}, "notes": []}
T0 = time.time()


def note(msg):
    OUT["notes"].append(f"{time.time() - T0:7.2f}s {msg}")


def bench(fn, iters=50, warm=12):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def mk(shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g).to(dtype=dtype)
    go = torch.randn(shape, generator=g).to(dtype=dtype)
    return go.npu(), x.npu()


REF = torch.ops.aten.hardshrink_backward

SHAPES = [
    ("f16_1k", [1024], torch.float16, 1729),
    ("f16_1m", [1048576], torch.float16, 1731),
    ("f16_4k2", [4096, 4096], torch.float16, 1732),
    ("f32_4k2", [4096, 4096], torch.float32, 1740),
]

DATA = {}
for name, shape, dt, seed in SHAPES:
    go, x = mk(shape, dt, seed)
    DATA[name] = (go, x)
    REF(go, x, 0.5)
    OUT["cases"][name] = {"aten_ms": bench(lambda: REF(go, x, 0.5), iters=40, warm=10)}
    note(f"aten {name} done")

sys.path.insert(0, os.getcwd())
try:
    import importlib

    seedmod = importlib.import_module("tmp.main")
    OUT["seed_import"] = "ok"
except Exception:
    seedmod = None
    OUT["seed_import"] = traceback.format_exc()[-2000:]

if seedmod is not None:
    for name, _, _, _ in SHAPES:
        go, x = DATA[name]
        OUT["cases"][name]["seed_ms"] = bench(lambda: seedmod.run(go, x, 0.5))
        note(f"seed {name} done")
    g1 = torch.randn(1, dtype=torch.float16).npu()
    x1 = torch.randn(1, dtype=torch.float16).npu()
    OUT["cases"]["f16_1elem_seed_ms"] = bench(lambda: seedmod.run(g1, x1, 0.5))

g, x = DATA["f16_1k"]
OUT["host"]["empty_like_ms"] = bench(lambda: torch.empty_like(g), iters=200)
OUT["host"]["current_stream_ms"] = bench(lambda: torch.npu.current_stream(), iters=200)
OUT["host"]["contiguous_ms"] = bench(lambda: g.contiguous(), iters=200)
OUT["host"]["reshape_as_ms"] = bench(lambda: g.reshape_as(g), iters=200)
dev = torch.npu.current_device()
try:
    raw = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
    OUT["host"]["has_raw_stream"] = raw is not None
    if raw is not None:
        OUT["host"]["raw_stream_ms"] = bench(lambda: raw(dev), iters=200)
        OUT["host"]["raw_equals_current"] = int(raw(dev)) == int(
            torch.npu.current_stream().npu_stream
        )
except Exception:
    OUT["host"]["raw_probe"] = traceback.format_exc()[-1000:]


@triton.jit
def _k_nomask(go_ptr, x_ptr, out_ptr, lambd, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    go = tl.load(go_ptr + offs)
    xx = tl.load(x_ptr + offs)
    r = tl.where(tl.abs(xx) <= lambd, tl.zeros(go.shape, go.dtype), go)
    tl.store(out_ptr + offs, r)


@triton.jit
def _k_masked(go_ptr, x_ptr, out_ptr, n, lambd, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    go = tl.load(go_ptr + offs, mask=m, other=0)
    xx = tl.load(x_ptr + offs, mask=m, other=0)
    r = tl.where(tl.abs(xx) <= lambd, tl.zeros(go.shape, go.dtype), go)
    tl.store(out_ptr + offs, r, mask=m)


VARIANTS = [
    ("f16_4k2", 8192),
    ("f16_4k2", 16384),
    ("f32_4k2", 4096),
    ("f16_1m", 8192),
]

for case, block in VARIANTS:
    go, x = DATA[case]
    nel = go.numel()
    out = torch.empty_like(go)
    grid = (triton.cdiv(nel, block),)
    tag = f"nomask_{block}"
    try:
        _k_nomask[grid](go, x, out, 0.5, BLOCK=block)
        torch.npu.synchronize()
        OUT["cases"][case][f"{tag}_bitmatch"] = bool(
            torch.equal(out, REF(go, x, 0.5))
        )
        OUT["cases"][case][f"{tag}_ms"] = bench(
            lambda: _k_nomask[grid](go, x, out, 0.5, BLOCK=block)
        )
        ck = _k_nomask.warmup(go, x, out, 0.5, BLOCK=block, grid=(1,))
        launcher = ck.run
        OUT["launcher"]["type"] = str(type(launcher))
        OUT["launcher"]["dir"] = [a for a in dir(launcher) if not a.startswith("__")]
        try:
            OUT["launcher"]["sig"] = str(inspect.signature(launcher.launch))
        except Exception:
            OUT["launcher"]["sig"] = traceback.format_exc()[-500:]
        OUT["launcher"]["packed_metadata"] = repr(ck.packed_metadata)[:300]
        try:
            md = launcher.metadata
            OUT["launcher"]["metadata"] = {
                k: str(getattr(md, k))[:200]
                for k in ("num_warps", "num_stages", "name", "shared")
                if hasattr(md, k)
            }
        except Exception:
            pass
        rawf = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
        L, fn, pm, gsz = launcher.launch, ck.function, ck.packed_metadata, grid[0]

        def direct_const():
            L(gsz, 1, 1, rawf(dev), fn, pm, None, None, None,
              go.data_ptr(), x.data_ptr(), out.data_ptr(), 0.5, block)

        def direct_none():
            L(gsz, 1, 1, rawf(dev), fn, pm, None, None, None,
              go.data_ptr(), x.data_ptr(), out.data_ptr(), 0.5, None)

        for t, f in (("const", direct_const), ("none_const", direct_none)):
            if rawf is None:
                OUT["launcher"][f"direct_{t}"] = "no raw stream api"
                continue
            try:
                f()
                torch.npu.synchronize()
                OUT["cases"][case][f"direct_{t}_{block}_ms"] = bench(f)
                OUT["launcher"][f"direct_{t}"] = "ok"
            except Exception:
                OUT["launcher"][f"direct_{t}"] = traceback.format_exc()[-700:]
        note(f"variant {case}/{block} done")
    except Exception:
        OUT["cases"][case][f"{tag}_err"] = traceback.format_exc()[-700:]
        note(f"variant {case}/{block} FAILED")

@triton.jit
def _k_persist(go_ptr, x_ptr, out_ptr, ntiles, lambd, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    for t in tl.range(pid, ntiles, npg):
        offs = t * BLOCK + tl.arange(0, BLOCK)
        go = tl.load(go_ptr + offs)
        xx = tl.load(x_ptr + offs)
        r = tl.where(tl.abs(xx) <= lambd, tl.zeros(go.shape, go.dtype), go)
        tl.store(out_ptr + offs, r)


go, x = DATA["f16_4k2"]
nel = go.numel()
out = torch.empty_like(go)
block, nprog = 8192, 20
ntiles = triton.cdiv(nel, block)
try:
    _k_persist[(nprog,)](go, x, out, ntiles, 0.5, BLOCK=block)
    torch.npu.synchronize()
    OUT["cases"]["f16_4k2"][f"persist_{block}_{nprog}_bitmatch"] = bool(
        torch.equal(out, REF(go, x, 0.5))
    )
    OUT["cases"]["f16_4k2"][f"persist_{block}_{nprog}_ms"] = bench(
        lambda: _k_persist[(nprog,)](go, x, out, ntiles, 0.5, BLOCK=block)
    )
except Exception:
    OUT["cases"]["f16_4k2"][f"persist_{block}_{nprog}_err"] = (
        traceback.format_exc()[-700:]
    )
note("persist done")

try:
    edge = [0.0, -0.0, float("nan"), float("inf"), -float("inf"), -1.0, 1.0]
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        xx = torch.randn(1025)
        xx[:7] = torch.tensor(edge)
        xx = xx.to(dt)
        gg = torch.randn(1025).to(dt)
        xn, gn = xx.npu(), gg.npu()
        ref = REF(gn, xn, 0.5)
        out = torch.empty_like(gn)
        _k_masked[(2,)](gn, xn, out, 1025, 0.5, BLOCK=1024)
        torch.npu.synchronize()
        OUT["numerics"][str(dt)] = {
            "bit_equal": bool(torch.equal(out, ref)),
            "ref_head": [float(v) for v in ref[:9].float().cpu()],
            "cand_head": [float(v) for v in out[:9].float().cpu()],
        }
except Exception:
    OUT["numerics"]["err"] = traceback.format_exc()[-1500:]

with open(os.path.join(ART, "probe.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print(json.dumps(OUT, indent=1)[:12000])
