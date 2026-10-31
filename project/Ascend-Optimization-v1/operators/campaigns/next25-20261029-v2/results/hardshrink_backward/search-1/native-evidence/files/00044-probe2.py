"""Bounded target-side probe #2 for hardshrink_backward (<=120 s).

Focus: (a) output-allocation cost variants, (b) tile / num_warps tuning for the
three 16.7M-element cases through the direct compiled launcher, (c) the true
per-call floor for the small (1024-element) case, (d) direct-launcher numerics.
"""
import json
import os
import time
import traceback

import torch
import torch_npu
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
os.makedirs(ART, exist_ok=True)
OUT = {"alloc": {}, "big": {}, "small": {}, "numerics": {}, "notes": []}
T0 = time.time()


def note(m):
    OUT["notes"].append(f"{time.time() - T0:7.2f}s {m}")


def bench(fn, iters=50, warm=12):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


REF = torch.ops.aten.hardshrink_backward
dev = torch.npu.current_device()
rawf = torch_npu._C._npu_getCurrentRawStreamNoWait


def mk(n, dt, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, generator=g).to(dt).npu()
    go = torch.randn(n, generator=g).to(dt).npu()
    return go, x


@triton.jit
def _k_nomask(go_ptr, x_ptr, out_ptr, lambd, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    go = tl.load(go_ptr + offs)
    xx = tl.load(x_ptr + offs)
    r = tl.where(tl.abs(xx) <= lambd, tl.zeros(go.shape, go.dtype), go)
    tl.store(out_ptr + offs, r)


def make_direct(go, x, out, block, warps):
    nel = go.numel()
    grid = (nel + block - 1) // block
    ck = _k_nomask.warmup(
        go, x, out, 0.5, BLOCK=block, num_warps=warps, grid=(1,)
    )
    L, fn, pm = ck.run.launch, ck.function, ck.packed_metadata
    gp, xp, op = go.data_ptr(), x.data_ptr(), out.data_ptr()

    def call():
        L(grid, 1, 1, rawf(dev), fn, pm, None, None, None, gp, xp, op, 0.5, block)

    return call, ck


# ---------------------------------------------------------------- allocation
g1k = torch.randn(1024, dtype=torch.float16).npu()
OUT["alloc"]["empty_like"] = bench(lambda: torch.empty_like(g1k), iters=300)
OUT["alloc"]["empty_shape"] = bench(
    lambda: torch.empty(g1k.shape, dtype=g1k.dtype, device=g1k.device), iters=300
)
OUT["alloc"]["new_empty"] = bench(lambda: g1k.new_empty(g1k.shape), iters=300)
OUT["alloc"]["empty_like_cf"] = bench(
    lambda: torch.empty_like(g1k, memory_format=torch.contiguous_format), iters=300
)
OUT["alloc"]["aten_empty_like"] = bench(
    lambda: torch.ops.aten.empty_like.default(g1k), iters=300
)
OUT["alloc"]["empty_1d_int"] = bench(
    lambda: torch.empty(1024, dtype=g1k.dtype, device=g1k.device), iters=300
)
note("alloc done")

# ---------------------------------------------------------------- big cases
BIG = [
    ("f16", 4096 * 4096, torch.float16, 1732, [(16384, 4), (16384, 8), (32768, 4)]),
    ("bf16", 4096 * 4096, torch.bfloat16, 1736, [(16384, 4), (32768, 4)]),
    ("f32", 4096 * 4096, torch.float32, 1740, [(8192, 4), (16384, 4), (8192, 8)]),
]
for tag, nel, dt, seed, cfgs in BIG:
    go, x = mk(nel, dt, seed)
    ref = REF(go, x, 0.5)
    out = torch.empty_like(go)
    OUT["big"][tag] = {"aten_ms": bench(lambda: REF(go, x, 0.5), iters=40, warm=10)}
    for block, warps in cfgs:
        key = f"b{block}_w{warps}"
        try:
            call, ck = make_direct(go, x, out, block, warps)
            call()
            torch.npu.synchronize()
            OUT["big"][tag][key + "_bitmatch"] = bool(torch.equal(out, ref))
            OUT["big"][tag][key + "_ms"] = bench(call)
        except Exception:
            OUT["big"][tag][key + "_err"] = traceback.format_exc()[-400:]
        note(f"big {tag} {key} done")

# ---------------------------------------------------------------- small floor
go, x = mk(1024, torch.float16, 1729)
out = torch.empty_like(go)
try:
    call, _ck = make_direct(go, x, out, 1024, 4)
    call()
    torch.npu.synchronize()
    OUT["small"]["direct_ms"] = bench(call)
    OUT["small"]["aten_ms"] = bench(lambda: REF(go, x, 0.5))
    OUT["small"]["bitmatch"] = bool(torch.equal(out, REF(go, x, 0.5)))


    def full():
        o = torch.empty_like(go)
        call_ptr = o.data_ptr()
        L, fn, pm, gp, xp = (_ck.run.launch, _ck.function, _ck.packed_metadata,
                             go.data_ptr(), x.data_ptr())
        L(1, 1, 1, rawf(dev), fn, pm, None, None, None, gp, xp, call_ptr, 0.5, 1024)
        return o

    def full_alloc_only():
        o = torch.empty_like(go)
        L, fn, pm, gp, xp = (_ck.run.launch, _ck.function, _ck.packed_metadata,
                             go.data_ptr(), x.data_ptr())
        L(1, 1, 1, rawf(dev), fn, pm, None, None, None, gp, xp, o.data_ptr(), 0.5, 1024)
        return o

    OUT["small"]["alloc_plus_launch_ms"] = bench(full_alloc_only)
except Exception:
    OUT["small"]["err"] = traceback.format_exc()[-800:]
note("small done")

with open(os.path.join(ART, "probe2.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print(json.dumps(OUT, indent=1)[:9000])
