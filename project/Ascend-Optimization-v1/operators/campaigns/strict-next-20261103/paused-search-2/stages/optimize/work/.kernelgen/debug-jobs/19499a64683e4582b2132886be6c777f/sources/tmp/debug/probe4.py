"""Probe 4 (diagnostic): split the persistent-loop cost into memory vs math.

Probe 3 established, at 16.7M elements through the compiled launcher:
  fp16 B8192 grid40  in-place copy 0.0237 ms | series+where 0.0596 ms
  fp32 B4096 grid40  in-place copy 0.0454 ms | series+where 0.1069 ms
so the value arithmetic, not the DMA volume, is the dominant term, and two
tiles per iteration with hoisted loads already recovers ~14% (fp16 0.0596 ->
0.0510).  This probe attributes the arithmetic and checks whether more tiles in
flight improve both the copy floor and the math-bound case.

usage: python probe4.py float16
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
LOG2E = 1.4426950408889634


@triton.jit
def _val(x, MODE: tl.constexpr, TINY: tl.constexpr):
    if MODE == 0:
        return x                                    # copy
    elif MODE == 1:
        return tl.exp(x) - 1.0                      # expm1 native
    elif MODE == 2:
        tiny = tl.full((1,), TINY, x.dtype)
        return tl.where(tl.abs(x) < tiny, x * (1.0 + 0.5 * x),
                        tl.exp(x) - 1.0)            # current candidate
    elif MODE == 3:
        return x * (1.0 + 0.5 * x)                  # series only, no exp
    elif MODE == 4:
        return tl.exp(x)                            # exp only
    elif MODE == 5:
        return tl.exp2(x * LOG2E) - 1.0             # exp2 form
    elif MODE == 6:
        f = x.to(tl.float32)
        return (tl.exp(f) - 1.0).to(x.dtype)        # exp in fp32
    elif MODE == 7:
        f = x.to(tl.float32)
        return (tl.exp2(f * LOG2E) - 1.0).to(x.dtype)
    tiny = tl.full((1,), TINY, x.dtype)
    return tl.where(tl.abs(x) < tiny, x, tl.exp(x) - 1.0)


@triton.jit(do_not_specialize=["n"])
def _u1(a_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr, TINY: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for start in range(tl.program_id(0) * BLOCK, n, step):
        off = start + lane
        m = off < n
        x = tl.load(a_ptr + off, mask=m, other=0.0)
        tl.store(a_ptr + off, _val(x, MODE, TINY), mask=m)


@triton.jit(do_not_specialize=["n"])
def _u2(a_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr, TINY: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * (2 * BLOCK)
    for start in range(tl.program_id(0) * (2 * BLOCK), n, step):
        o0 = start + lane
        o1 = start + BLOCK + lane
        m0 = o0 < n
        m1 = o1 < n
        x0 = tl.load(a_ptr + o0, mask=m0, other=0.0)
        x1 = tl.load(a_ptr + o1, mask=m1, other=0.0)
        tl.store(a_ptr + o0, _val(x0, MODE, TINY), mask=m0)
        tl.store(a_ptr + o1, _val(x1, MODE, TINY), mask=m1)


@triton.jit(do_not_specialize=["n"])
def _u4(a_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr, TINY: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * (4 * BLOCK)
    for start in range(tl.program_id(0) * (4 * BLOCK), n, step):
        o0 = start + lane
        o1 = start + BLOCK + lane
        o2 = start + 2 * BLOCK + lane
        o3 = start + 3 * BLOCK + lane
        m0 = o0 < n
        m1 = o1 < n
        m2 = o2 < n
        m3 = o3 < n
        x0 = tl.load(a_ptr + o0, mask=m0, other=0.0)
        x1 = tl.load(a_ptr + o1, mask=m1, other=0.0)
        x2 = tl.load(a_ptr + o2, mask=m2, other=0.0)
        x3 = tl.load(a_ptr + o3, mask=m3, other=0.0)
        tl.store(a_ptr + o0, _val(x0, MODE, TINY), mask=m0)
        tl.store(a_ptr + o1, _val(x1, MODE, TINY), mask=m1)
        tl.store(a_ptr + o2, _val(x2, MODE, TINY), mask=m2)
        tl.store(a_ptr + o3, _val(x3, MODE, TINY), mask=m3)


KEN = {1: _u1, 2: _u2, 4: _u4}


def bench(fn, iters=40, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


N = 4096 * 4096
DT = {"float16": (torch.float16, 0.125, 8192),
      "bfloat16": (torch.bfloat16, 0.125, 8192),
      "float32": (torch.float32, 0.0078125, 4096)}

for name in (sys.argv[1] if len(sys.argv) > 1 else "float16").split(","):
    dtype, tiny, block = DT[name]
    rec = REC["variants"].setdefault(name, {})
    g = torch.Generator().manual_seed(1234)
    a = torch.randn(N, generator=g).to(dtype).to("npu:0")
    ptr = a.data_ptr()
    st = _RAW(0) if _RAW is not None else torch.npu.current_stream(0).npu_stream

    plan = [(1, 0), (2, 0), (4, 0), (1, 2), (2, 2), (4, 2), (1, 1), (2, 1),
            (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 5), (2, 6)]
    for unroll, mode in plan:
        tag = f"u{unroll}_M{mode}"
        try:
            ck = KEN[unroll].warmup(a, N, BLOCK=block, MODE=mode, TINY=tiny,
                                    grid=(1,))
            launch, fn, meta = ck.run.launch, ck.function, ck.packed_metadata
            grid = 40
            rec[tag] = round(bench(
                lambda: launch(grid, 1, 1, st, fn, meta, None, None, None,
                               ptr, N, block, mode, tiny)), 5)
        except Exception as exc:
            rec[tag] = "ERR:" + repr(exc)[:90]

with open(os.path.join(ART, "probe4.json"), "w") as fh:
    json.dump(REC, fh, indent=1, sort_keys=True)
print(json.dumps(REC, indent=1, sort_keys=True))
