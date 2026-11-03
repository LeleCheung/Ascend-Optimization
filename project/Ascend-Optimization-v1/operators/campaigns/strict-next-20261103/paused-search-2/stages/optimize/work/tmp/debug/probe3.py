"""Probe 3 (diagnostic): persistent-loop device-time map through the compiled
launcher (host ~10 us, validated to reproduce the eval latencies within 2%).

Knobs per element type: tile size, one vs two tiles per iteration with loads
hoisted before stores, compile-time grid (provable alignment), and a mask-free
main loop with one masked tail tile.

usage: python probe3.py float16,float32
"""

import json
import os
import sys
import time
import traceback

import torch
import torch_npu
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
torch.npu.set_device(0)
REC = {"variants": {}, "notes": []}
_RAW = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)


@triton.jit
def _val(x, MODE: tl.constexpr, TINY: tl.constexpr):
    if MODE == 0:
        return x
    elif MODE == 1:
        return tl.exp(x) - 1.0
    tiny = tl.full((1,), TINY, x.dtype)
    return tl.where(tl.abs(x) < tiny, x * (1.0 + 0.5 * x), tl.exp(x) - 1.0)


@triton.jit(do_not_specialize=["n"])
def _p1(a_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr,
        TINY: tl.constexpr, GRIDC: tl.constexpr, NOMASK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    if GRIDC > 0:
        step = GRIDC * BLOCK
    else:
        step = tl.num_programs(0) * BLOCK
    base = tl.program_id(0) * BLOCK
    if NOMASK:
        full = (n // step) * step
        for start in range(base, full, step):
            off = start + lane
            x = tl.load(a_ptr + off)
            tl.store(a_ptr + off, _val(x, MODE, TINY))
        off = full + base + lane
        m = off < n
        x = tl.load(a_ptr + off, mask=m, other=0.0)
        tl.store(a_ptr + off, _val(x, MODE, TINY), mask=m)
    else:
        for start in range(base, n, step):
            off = start + lane
            m = off < n
            x = tl.load(a_ptr + off, mask=m, other=0.0)
            tl.store(a_ptr + off, _val(x, MODE, TINY), mask=m)


@triton.jit(do_not_specialize=["n"])
def _p2(a_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr,
        TINY: tl.constexpr, GRIDC: tl.constexpr, NOMASK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    if GRIDC > 0:
        step = GRIDC * (2 * BLOCK)
    else:
        step = tl.num_programs(0) * (2 * BLOCK)
    base = tl.program_id(0) * (2 * BLOCK)
    if NOMASK:
        full = (n // step) * step
        for start in range(base, full, step):
            o0 = start + lane
            o1 = start + BLOCK + lane
            x0 = tl.load(a_ptr + o0)
            x1 = tl.load(a_ptr + o1)
            tl.store(a_ptr + o0, _val(x0, MODE, TINY))
            tl.store(a_ptr + o1, _val(x1, MODE, TINY))
        o0 = full + base + lane
        o1 = o0 + BLOCK
        m0 = o0 < n
        m1 = o1 < n
        x0 = tl.load(a_ptr + o0, mask=m0, other=0.0)
        x1 = tl.load(a_ptr + o1, mask=m1, other=0.0)
        tl.store(a_ptr + o0, _val(x0, MODE, TINY), mask=m0)
        tl.store(a_ptr + o1, _val(x1, MODE, TINY), mask=m1)
    else:
        for start in range(base, n, step):
            o0 = start + lane
            o1 = start + BLOCK + lane
            m0 = o0 < n
            m1 = o1 < n
            x0 = tl.load(a_ptr + o0, mask=m0, other=0.0)
            x1 = tl.load(a_ptr + o1, mask=m1, other=0.0)
            tl.store(a_ptr + o0, _val(x0, MODE, TINY), mask=m0)
            tl.store(a_ptr + o1, _val(x1, MODE, TINY), mask=m1)


def entry_of(kernel, t, args, kwargs, grid=(1,)):
    ck = kernel.warmup(t, *args, grid=grid, **kwargs)
    return ck.run.launch, ck.function, ck.packed_metadata


def bench(fn, iters=40, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def call_of(e, ptr, n, grid, cargs):
    launch, fn, meta = e
    st = _RAW(0) if _RAW is not None else torch.npu.current_stream(0).npu_stream

    def c():
        launch(grid, 1, 1, st, fn, meta, None, None, None, ptr, n, *cargs)
    return c


N = 4096 * 4096
CTX = {}


def sweep(dtype, kind, blocks, modes, grids, tiny, nomask, iters=40):
    key = f"{str(dtype).replace('torch.', '')}"
    rec = REC["variants"].setdefault(key, {})
    g = torch.Generator().manual_seed(1234)
    a = torch.randn(N, generator=g).to(dtype).to("npu:0")
    ptr = a.data_ptr()
    CTX["a"] = a
    kernel = _p1 if kind == 1 else _p2
    for block in blocks:
        for mask in nomask:
            for mode in (modes if block == blocks[0] else modes[-1:]):
                tag = f"p{kind}_B{block}_M{mode}_N{mask}"
                try:
                    e = entry_of(kernel, a, (N,), dict(
                        BLOCK=block, MODE=mode, TINY=tiny, GRIDC=0,
                        NOMASK=mask))
                except Exception as exc:
                    rec[tag] = "COMPILE_ERR:" + repr(exc)[:100]
                    continue
                for grid in grids:
                    try:
                        rec[f"{tag}_G{grid}"] = round(bench(
                            call_of(e, ptr, N, grid,
                                    (block, mode, tiny, 0, mask)), iters), 5)
                    except Exception as exc:
                        rec[f"{tag}_G{grid}"] = "ERR:" + repr(exc)[:100]
        # compile-time grid constant for the fastest-moving config
        try:
            gc = grids[0]
            e2 = entry_of(kernel, a, (N,), dict(
                BLOCK=block, MODE=modes[-1], TINY=tiny, GRIDC=gc,
                NOMASK=nomask[-1]))
            rec[f"p{kind}_B{block}_M{modes[-1]}_N{nomask[-1]}_GC{gc}"] = round(
                bench(call_of(e2, ptr, N, gc,
                              (block, modes[-1], tiny, gc, nomask[-1])),
                      iters), 5)
        except Exception as exc:
            rec[f"p{kind}_B{block}_GC"] = "ERR:" + repr(exc)[:100]


DT = {"float16": (torch.float16, 0.125), "bfloat16": (torch.bfloat16, 0.125),
      "float32": (torch.float32, 0.0078125)}
want = (sys.argv[1] if len(sys.argv) > 1 else "float16").split(",")

for name in want:
    dtype, tiny = DT[name]
    plan = {
        "float16": ((8192, 16384, 32768), (40, 64, 80, 160)),
        "bfloat16": ((8192, 16384), (40, 64, 80, 160)),
        "float32": ((4096, 8192, 16384), (40, 64, 80, 160)),
    }[name]
    try:
        sweep(dtype, 1, plan[0], (0, 2), plan[1], tiny, (0, 1))
        sweep(dtype, 2, plan[0][:2], (2,), plan[1], tiny, (0, 1))
    except Exception:
        REC["notes"].append(f"{name}: " + traceback.format_exc()[-400:])

with open(os.path.join(ART, "probe3.json"), "w") as fh:
    json.dump(REC, fh, indent=1, sort_keys=True)
print(json.dumps(REC, indent=1, sort_keys=True))
