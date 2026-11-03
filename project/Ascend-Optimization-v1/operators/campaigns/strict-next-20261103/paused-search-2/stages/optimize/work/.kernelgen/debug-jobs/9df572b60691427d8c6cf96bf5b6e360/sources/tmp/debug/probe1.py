"""Bounded target-side probe: where does the 4096x4096 expm1_ device time go?

Measures device time of small kernel variants (all strict Triton) for the three
element types at the large timing shapes, plus one host-floor shape, and dumps
whatever IR levels the compiler exposes so the DMA lowering can be inspected.

Diagnostic only: no formal correctness / speedup claim is derived from here.
"""

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

REC = {"variants": {}, "ir": {}, "notes": []}


# ---------------------------------------------------------------- kernels ----
@triton.jit
def _k_flat_copy(a_ptr, N: tl.constexpr, BLOCK: tl.constexpr, MASK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if MASK:
        m = off < N
        x = tl.load(a_ptr + off, mask=m, other=0.0)
        tl.store(a_ptr + off, x, mask=m)
    else:
        x = tl.load(a_ptr + off)
        tl.store(a_ptr + off, x)


@triton.jit
def _k_flat_val(a_ptr, N: tl.constexpr, BLOCK: tl.constexpr,
                MODE: tl.constexpr, TINY: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(a_ptr + off)
    if MODE == 0:                       # branch-free, native element type
        y = tl.exp(x) - 1.0
    elif MODE == 1:                     # current series + where, native type
        tiny = tl.full((1,), TINY, x.dtype)
        y = tl.where(tl.abs(x) < tiny, x * (1.0 + 0.5 * x), tl.exp(x) - 1.0)
    elif MODE == 2:                     # branch-free, widened to fp32
        y = (tl.exp(x.to(tl.float32)) - 1.0).to(x.dtype)
    else:                               # current series + where, fp32 math
        f = x.to(tl.float32)
        y = tl.where(tl.abs(f) < TINY, f * (1.0 + 0.5 * f), tl.exp(f) - 1.0)
        y = y.to(x.dtype)
    tl.store(a_ptr + off, y)


@triton.jit(do_not_specialize=["n"])
def _k_loop2(a_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr,
             TINY: tl.constexpr):
    """Candidate shape: persistent loop, two tiles, per-tile masks."""
    lane = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * (2 * BLOCK)
    for start in range(tl.program_id(0) * (2 * BLOCK), n, step):
        off0 = start + lane
        off1 = start + BLOCK + lane
        m0 = off0 < n
        m1 = off1 < n
        x0 = tl.load(a_ptr + off0, mask=m0, other=0.0)
        x1 = tl.load(a_ptr + off1, mask=m1, other=0.0)
        if MODE == 0:
            y0 = tl.exp(x0) - 1.0
            y1 = tl.exp(x1) - 1.0
        else:
            tiny = tl.full((1,), TINY, x0.dtype)
            y0 = tl.where(tl.abs(x0) < tiny, x0 * (1.0 + 0.5 * x0),
                          tl.exp(x0) - 1.0)
            y1 = tl.where(tl.abs(x1) < tiny, x1 * (1.0 + 0.5 * x1),
                          tl.exp(x1) - 1.0)
        tl.store(a_ptr + off0, y0, mask=m0)
        tl.store(a_ptr + off1, y1, mask=m1)


# ------------------------------------------------------------- utilities ----
def _cores():
    from triton.runtime import driver
    try:
        p = driver.active.utils.get_device_properties(0)
        return int(dict(p).get("num_vectorcore", 0) or 0)
    except Exception:
        return 0


def bench(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def tensor(dtype, numel):
    g = torch.Generator().manual_seed(1234)
    base = torch.randn(numel, generator=g).to(dtype)
    return base.to(DEV)


def run_case(dtype, numel):
    key = f"{str(dtype).replace('torch.', '')}-{numel}"
    cores = CORES
    a = tensor(dtype, numel)
    TINY = 0.0078125 if dtype == torch.float32 else 0.125
    res = {}
    blocks = {torch.float16: (8192, 2048), torch.bfloat16: (8192, 2048),
              torch.float32: (4096, 1024)}[dtype]
    blk = blocks[0]
    iters = 30 if numel > (1 << 20) else 200

    def flat(kern, fn_kwargs, grid, **ir):
        def f():
            kern[grid](a, numel, **fn_kwargs)
        try:
            res[fn_kwargs.get("_name", kern.fn.__name__)] = round(bench(f, iters), 5)
        except Exception as e:
            res[fn_kwargs.get("_name", kern.fn.__name__)] = "ERR:" + str(e)[:120]

    # 1. masked vs unmasked in-place copy at exactly the same tile
    for mask in (True, False):
        g = ((numel + blk - 1) // blk,) if mask else (numel // blk,)
        name = "flatcopy_mask" if mask else "flatcopy_nomask"
        try:
            res[name] = round(bench(
                lambda g=g, mask=mask: _k_flat_copy[g](
                    a, numel, BLOCK=blk, MASK=mask), iters), 5)
        except Exception as e:
            res[name] = "ERR:" + str(e)[:120]

    # 2. value functions on mask-free flat tiles
    for mode, tag in ((0, "val_bfree_native"), (1, "val_series_native"),
                      (2, "val_bfree_fp32"), (3, "val_series_fp32")):
        try:
            res[tag] = round(bench(
                lambda mode=mode: _k_flat_val[(numel // blk,)](
                    a, numel, BLOCK=blk, MODE=mode, TINY=TINY), iters), 5)
        except Exception as e:
            res[tag] = "ERR:" + str(e)[:120]

    # 3. the candidate's own persistent two-tile loop, masked, both values
    for mode, tag in ((0, "loop2_bfree"), (1, "loop2_series")):
        try:
            res[tag] = round(bench(
                lambda mode=mode: _k_loop2[(cores or 40,)](
                    a, numel, BLOCK=blk, MODE=mode, TINY=TINY), iters), 5)
        except Exception as e:
            res[tag] = "ERR:" + str(e)[:120]

    REC["variants"][key] = res
    return a


def dump_ir():
    """Which IR levels exist, and what does a masked vs unmasked load look like."""
    a = tensor(torch.float32, 4096 * 1024)
    for tag, kwargs, grid in (
            ("copy_mask", dict(N=4096 * 1024, BLOCK=4096, MASK=True), (1024,)),
            ("copy_nomask", dict(N=4096 * 1024, BLOCK=4096, MASK=False), (1024,)),
            ("val_bfree", dict(N=4096 * 1024, BLOCK=4096, MODE=0, TINY=0.0078125),
             (1024,)),
    ):
        try:
            ck = _k_flat_copy.warmup(a, grid=grid, **kwargs) if tag.startswith(
                "copy") else _k_flat_val.warmup(a, grid=grid, **kwargs)
            asm = getattr(ck, "asm", None) or {}
            REC["ir"][tag] = {k: len(v) if isinstance(v, str) else str(type(v))
                              for k, v in asm.items()}
            for k, v in asm.items():
                if isinstance(v, str) and k.lower() in ("ttir", "ttgir", "llir",
                                                        "mlir", "npuir", "triton"):
                    fn = os.path.join(ART, f"{tag}.{k}.txt")
                    with open(fn, "w") as fh:
                        fh.write(v)
        except Exception as e:
            REC["ir"][tag] = "ERR:" + str(e)[:200]


CORES = _cores()
REC["cores"] = CORES
REC["versions"] = {
    "torch": torch.__version__,
    "triton": triton.__version__,
}

try:
    dump_ir()
except Exception:
    REC["ir"]["_dump_failed"] = traceback.format_exc()[-500:]

for dtype in (torch.float16, torch.bfloat16, torch.float32):
    for numel in (4096 * 4096,):
        try:
            run_case(dtype, numel)
        except Exception:
            REC["variants"][f"{dtype}-{numel}"] = traceback.format_exc()[-400:]

# small-shape host floor
for dtype in (torch.float32,):
    a = tensor(dtype, 1024)
    try:
        REC["host_floor_us_1024_fp32"] = round(bench(
            lambda: _k_flat_copy[(1,)](a, 1024, BLOCK=1024, MASK=True), 500), 5) * 1e3
    except Exception as e:
        REC["host_floor_us_1024_fp32"] = "ERR:" + str(e)[:120]

with open(os.path.join(ART, "probe1.json"), "w") as fh:
    json.dump(REC, fh, indent=1, sort_keys=True)
print(json.dumps(REC, indent=1, sort_keys=True))
