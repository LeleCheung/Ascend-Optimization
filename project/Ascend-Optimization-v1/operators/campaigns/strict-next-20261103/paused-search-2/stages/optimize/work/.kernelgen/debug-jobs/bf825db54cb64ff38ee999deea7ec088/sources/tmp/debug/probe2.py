"""Probe 2 (diagnostic): device-time map of the persistent-loop memory path.

Probe 1 showed that a one-tile-per-program flat grid is catastrophic (grid=4096
blocks: 0.080 ms fp16 / 0.157 ms fp32 for a pure in-place copy) and that the
value math is free on top of the copy in that configuration, so the limiting
factor is the memory/DMA path of the persistent loop, not the arithmetic.

This probe times the persistent-loop family through the *compiled launcher*
(the same path the candidate uses, ~10 us host instead of ~60 us for the JIT
dispatcher), varying tile size, grid size, unroll and whether the grid is a
compile-time constant (which makes the induction-variable alignment provable).
"""

import importlib.util
import json
import os
import time
import traceback

import torch
import torch_npu
import triton
import triton.language as tl

DEV = "npu:0"
ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
torch.npu.set_device(0)
REC = {"variants": {}, "ir_keys": {}, "notes": []}
_RAW = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None) or (
    lambda d: torch.npu.current_stream(d).npu_stream)


@triton.jit(do_not_specialize=["n"])
def _k_pers(a_ptr, n, BLOCK: tl.constexpr, UNROLL: tl.constexpr,
            MODE: tl.constexpr, TINY: tl.constexpr, GRIDC: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    if GRIDC > 0:
        step = GRIDC * (UNROLL * BLOCK)
        base = tl.program_id(0) * (UNROLL * BLOCK)
    else:
        step = tl.num_programs(0) * (UNROLL * BLOCK)
        base = tl.program_id(0) * (UNROLL * BLOCK)
    for start in range(base, n, step):
        for u in tl.static_range(UNROLL):
            off = start + u * BLOCK + lane
            m = off < n
            x = tl.load(a_ptr + off, mask=m, other=0.0)
            if MODE == 0:
                y = x
            elif MODE == 1:
                y = tl.exp(x) - 1.0
            else:
                tiny = tl.full((1,), TINY, x.dtype)
                y = tl.where(tl.abs(x) < tiny, x * (1.0 + 0.5 * x),
                             tl.exp(x) - 1.0)
            tl.store(a_ptr + off, y, mask=m)


def entry_of(kernel, ptr_tensor, args, kwargs, grid):
    ck = kernel.warmup(ptr_tensor, *args, grid=grid, **kwargs)
    launcher = ck.run
    return (launcher.launch, ck.function, ck.packed_metadata, ck)


def bench(call, iters, warmup=5):
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        call()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def tensor(dtype, numel):
    g = torch.Generator().manual_seed(1234)
    return torch.randn(numel, generator=g).to(dtype).to(DEV)


def make_call(e, n, grid, block, unroll, mode, tiny, gridc):
    launch, fn, meta = e[0], e[1], e[2]
    st = _RAW(0)

    def call():
        launch(grid, 1, 1, st, fn, meta, None, None, None,
               A_PTR, n, block, unroll, mode, tiny, gridc)
    return call


def sweep(dtype, blocks, modes, grids, tiny, iters=40):
    key = str(dtype).replace("torch.", "")
    res = {}
    global A_PTR
    a = tensor(dtype, N)
    A_PTR = a.data_ptr()
    for block in blocks:
        for mode in modes:
            tag = f"B{block}_M{mode}"
            try:
                e = entry_of(_k_pers, a, (N,),
                             dict(BLOCK=block, UNROLL=1, MODE=mode, TINY=tiny,
                                  GRIDC=0), (1,))
            except Exception as exc:
                res[tag] = "COMPILE_ERR:" + repr(exc)[:110]
                continue
            for grid in (grids + [0]):
                if grid == 0 and block != blocks[0]:
                    continue
                if grid == 0:
                    # compile-time grid constant: alignment provable
                    tagn = tag + "_GC%d" % grids[-1]
                    try:
                        e2 = entry_of(_k_pers, a, (N,),
                                      dict(BLOCK=block, UNROLL=1, MODE=mode,
                                           TINY=tiny, GRIDC=grids[-1]), (1,))
                        res[tagn] = round(bench(
                            make_call(e2, N, grids[-1], block, 1, mode,
                                      tiny, grids[-1]), iters), 5)
                    except Exception as exc:
                        res[tagn] = "ERR:" + repr(exc)[:110]
                    continue
                res[f"{tag}_G{grid}"] = round(bench(
                    make_call(e, N, grid, block, 1, mode, tiny, 0),
                    iters), 5)
        REC["variants"].setdefault(key, {}).update(res)
    return a


N = 4096 * 4096
A_PTR = 0
REC["n"] = N

# --- reference point: the shipped candidate's own run() through its fast path
try:
    spec = importlib.util.spec_from_file_location("cand_main", "tmp/main.py")
    cand = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cand)
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        a = tensor(dtype, N)
        cand.run(a)
        torch.npu.synchronize()
        REC["variants"].setdefault(
            str(dtype).replace("torch.", ""), {})["cand_run"] = round(
                bench(lambda a=a: cand.run(a), 40), 5)
except Exception:
    REC["notes"].append("cand_run failed: " + traceback.format_exc()[-300:])

# --- memory/DMA path map
for dtype, blocks, tiny in (
        (torch.float16, (8192, 16384, 32768), 0.125),
        (torch.bfloat16, (8192, 16384), 0.125),
        (torch.float32, (4096, 8192, 16384), 0.0078125)):
    try:
        sweep(dtype, blocks, (2, 0), (40, 80, 160), tiny)
    except Exception:
        REC["notes"].append(f"sweep {dtype}: " + traceback.format_exc()[-300:])

# --- IR of the persistent masked loop (what DMA does the backend emit?)
try:
    a = tensor(torch.float16, N)
    ck = _k_pers.warmup(a, N, BLOCK=8192, UNROLL=1, MODE=2, TINY=0.125,
                        GRIDC=0, grid=(1,))
    asm = getattr(ck, "asm", None) or {}
    REC["ir_keys"] = list(asm.keys())
    for k in ("source", "ttadapter", "ttir"):
        v = asm.get(k)
        if isinstance(v, str):
            with open(os.path.join(ART, f"pers.{k}.txt"), "w") as fh:
                fh.write(v)
except Exception:
    REC["notes"].append("ir dump: " + traceback.format_exc()[-300:])

with open(os.path.join(ART, "probe2.json"), "w") as fh:
    json.dump(REC, fh, indent=1, sort_keys=True)
print(json.dumps(REC, indent=1, sort_keys=True))
