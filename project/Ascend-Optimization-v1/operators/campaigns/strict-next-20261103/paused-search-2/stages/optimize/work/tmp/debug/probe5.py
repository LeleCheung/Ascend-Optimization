"""Probe 5 (diagnostic): fp32 and bf16 component costs, plus which exp forms exist.

fp16 findings (probe 4, 16.7M elements, B8192, grid 40, compiled launcher):
  copy 0.0222 | exp 0.0329 | exp-1 0.0295(u2) | series-only 0.0346 | current 0.0515
=> roughly 3.5-4.5 us per extra elementwise op; the fp16 hardware exp is cheap
(~2.5 ops) but the small-|x| series + abs/compare/select costs more than it saves.

usage: python probe5.py float32,bfloat16
"""

import json
import os
import sys
import time

import torch
import torch_npu
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
torch.npu.set_device(0)
REC = {"variants": {}, "errors": {}}
_RAW = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
LOG2E = 1.4426950408889634


@triton.jit
def _val(x, MODE: tl.constexpr, TINY: tl.constexpr):
    if MODE == 0:
        return x
    elif MODE == 1:
        return tl.exp(x) - 1.0
    elif MODE == 2:
        tiny = tl.full((1,), TINY, x.dtype)
        return tl.where(tl.abs(x) < tiny, x * (1.0 + 0.5 * x),
                        tl.exp(x) - 1.0)
    elif MODE == 3:
        return x * (1.0 + 0.5 * x)
    elif MODE == 4:
        return tl.exp(x)
    elif MODE == 5:
        return tl.math.exp2(x * LOG2E) - 1.0
    elif MODE == 6:
        return tl.exp2(x) - 1.0
    elif MODE == 7:
        return tl.where(x < 0.0, -1.0 + tl.exp(x), tl.exp(x) - 1.0)
    elif MODE == 8:
        # expm1 without any branch/select: exp is exact near zero only to 1 ulp
        return tl.exp(x) - 1.0 + x * 0.0
    return tl.exp(x) - 1.0


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


KEN = {1: _u1, 2: _u2}


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
PLAN = {"float32": [(1, 0), (1, 1), (1, 4), (1, 3), (1, 2), (2, 1), (2, 0),
                    (1, 5), (1, 6), (2, 2)],
        "bfloat16": [(1, 0), (1, 1), (1, 4), (1, 2), (2, 1), (2, 2)],
        "float16": [(1, 6), (1, 5), (2, 1)]}

for name in (sys.argv[1] if len(sys.argv) > 1 else "float32").split(","):
    dtype, tiny, block = DT[name]
    rec = REC["variants"].setdefault(name, {})
    g = torch.Generator().manual_seed(1234)
    a = torch.randn(N, generator=g).to(dtype).to("npu:0")
    ptr = a.data_ptr()
    st = _RAW(0) if _RAW is not None else torch.npu.current_stream(0).npu_stream
    for unroll, mode in PLAN[name]:
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
            rec[tag] = "ERR"
            REC["errors"][f"{name}_{tag}"] = repr(exc)[:600]

with open(os.path.join(ART, "probe5.json"), "w") as fh:
    json.dump(REC, fh, indent=1, sort_keys=True)
print(json.dumps(REC, indent=1, sort_keys=True))
