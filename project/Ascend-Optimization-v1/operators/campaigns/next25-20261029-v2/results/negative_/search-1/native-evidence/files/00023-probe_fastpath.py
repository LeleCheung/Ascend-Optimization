"""Build a low-overhead direct launcher path and re-measure sizes under it.

Diagnostic only. Writes the artifact incrementally so a device fault cannot
lose earlier stages.
"""
import json
import os
import time
import traceback

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

import triton
import triton.language as tl
from triton import knobs
from triton.runtime import driver as tdrv

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
OUT = os.path.join(ART, "fastpath_probe.json")
DEV = "npu"
res = {}
_raw = []


def flush():
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)


@triton.jit
def neg_flat(X, n, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(X + offs)
        tl.store(X + offs, -x)
    else:
        m = offs < n
        x = tl.load(X + offs, mask=m)
        tl.store(X + offs, -x, mask=m)


@triton.jit
def neg_gs(X, n, n_iter, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    step = npg * BLOCK
    base = pid * BLOCK
    for _ in range(n_iter):
        offs = base + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(X + offs, mask=m)
        tl.store(X + offs, -x, mask=m)
        base += step


def bench_wall(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


# ---------------- stage A: stream getters -------------------------------
stage = {}
try:
    cands = {
        "driver.get_current_stream(None)": lambda: tdrv.active.get_current_stream(None),
        "driver.get_current_stream(0)": lambda: tdrv.active.get_current_stream(0),
        "torch.npu.current_stream().npu_stream": lambda: torch.npu.current_stream().npu_stream,
        "C._npu_getCurrentStream(0)": lambda: torch_npu._C._npu_getCurrentStream(0),
        "C._npu_getCurrentStream()": lambda: torch_npu._C._npu_getCurrentStream(),
    }
    for k, fn in cands.items():
        try:
            fn()
            stage[k] = bench_wall(fn, 500)
        except Exception:
            stage[k] = traceback.format_exc()[-200:]
    import inspect
    import triton.backends.ascend.driver as _ad
    stage["get_backend_func_src"] = inspect.getsource(_ad.get_backend_func)[:1200]
    try:
        import triton.backends.ascend.utils as _au
        stage["utils_get_current_stream_src"] = inspect.getsource(_au)[:0] or "n/a"
    except Exception:
        pass
    stage["resolve"] = repr(tdrv.active.utils)
except Exception:
    stage["error"] = traceback.format_exc()[-800:]
res["stream_getters_ms"] = stage
flush()

# ---------------- stage B: fast launcher --------------------------------
x_small = torch.randn([1024], device=DEV, dtype=torch.float16)
ck = neg_flat.warmup(x_small, 1024, BLOCK=1024, EVEN=True, grid=(1,))
ck._init_handles()
launcher = ck._run  # NPULauncher instance
raw = launcher.launch  # generated C launch
res["raw_repr"] = repr(raw)[:200]
stream = tdrv.active.get_current_stream(None)
grid3 = (1, 1, 1)
enter = knobs.runtime.launch_enter_hook
exit_ = knobs.runtime.launch_exit_hook
res["hooks"] = [repr(enter)[:80], repr(exit_)[:80], bool(enter), bool(exit_)]

variants = {}


def mk_args(meta, hooks, s, tens, nn, blk, even):
    if hooks:
        return (grid3[0], grid3[1], grid3[2], s, ck.function, ck.packed_metadata, meta, enter, exit_,
                tens, nn, blk, even)
    return (grid3[0], grid3[1], grid3[2], s, ck.function, ck.packed_metadata, None, None, None,
            tens, nn, blk, even)


# B1: metadata None, hooks None
try:
    xx = torch.full([1024], 3.0, device=DEV, dtype=torch.float16)
    a0 = mk_args(None, False, stream, xx, 1024, 1024, True)
    raw(*a0)
    torch.npu.synchronize()
    ok = bool(torch.equal(xx, torch.full_like(xx, -3.0)))
    variants["no_meta_no_hooks"] = "ok" if ok else "ran but WRONG"
    if ok:
        res["t_no_meta_no_hooks"] = bench_wall(lambda: raw(*a0), 500)
except Exception:
    variants["no_meta_no_hooks"] = traceback.format_exc()[-400:]
res["variant_status"] = dict(variants)
flush()

# B2: hooks objects, metadata None
try:
    xx = torch.full([1024], 3.0, device=DEV, dtype=torch.float16)
    a1 = mk_args(None, True, stream, xx, 1024, 1024, True)
    raw(*a1)
    torch.npu.synchronize()
    ok = bool(torch.equal(xx, torch.full_like(xx, -3.0)))
    variants["hooksobj_no_meta"] = "ok" if ok else "ran but WRONG"
    if ok:
        res["t_hooks_no_meta"] = bench_wall(lambda: raw(*a1), 500)
except Exception:
    variants["hooksobj_no_meta"] = traceback.format_exc()[-400:]
res["variant_status"] = dict(variants)
flush()

# B3: full form incl. per-call launch_metadata LazyDict
try:
    xx = torch.full([1024], 3.0, device=DEV, dtype=torch.float16)
    lm = ck.launch_metadata(grid3, stream, xx, 1024, 1024, True)
    a2 = mk_args(lm, True, stream, xx, 1024, 1024, True)
    raw(*a2)
    torch.npu.synchronize()
    ok = bool(torch.equal(xx, torch.full_like(xx, -3.0)))
    variants["full_meta"] = "ok" if ok else "ran but WRONG"
    res["t_full_meta"] = bench_wall(lambda: raw(*a2), 500)
    res["t_launch_metadata_build"] = bench_wall(
        lambda: ck.launch_metadata(grid3, stream, xx, 1024, 1024, True), 500)
except Exception:
    variants["full_meta"] = traceback.format_exc()[-400:]
res["variant_status"] = dict(variants)
flush()

# B4: per-call arg build + stream fetch + raw call (realistic fast path)
try:
    good = variants.get("hooksobj_no_meta") == "ok" or variants.get("full_meta") == "ok"
    use_hooks = variants.get("hooksobj_no_meta") == "ok"
    use_meta = variants.get("full_meta") == "ok" and not use_hooks

    def fast_call(t, nn, blk, even, gx):
        s = tdrv.active.get_current_stream(None)
        if use_meta:
            m = ck.launch_metadata((gx, 1, 1), s, t, nn, blk, even)
        else:
            m = None
        if use_hooks:
            raw(gx, 1, 1, s, ck.function, ck.packed_metadata, m, enter, exit_, t, nn, blk, even)
        else:
            raw(gx, 1, 1, s, ck.function, ck.packed_metadata, None, None, None, t, nn, blk, even)

    xx = torch.randn([1024], device=DEV, dtype=torch.float16)
    exp = xx.clone()
    torch.neg_(exp)
    fast_call(xx, 1024, 1024, True, 1)
    torch.npu.synchronize()
    res["fast_call_ok"] = bool(torch.equal(xx, exp))
    res["t_fast_call_total"] = bench_wall(lambda: fast_call(xx, 1024, 1024, True, 1), 500)
    res["t_torch_neg_1024"] = bench_wall(lambda: torch.neg_(xx), 500)
except Exception:
    res["fast_call_err"] = traceback.format_exc()[-800:]
flush()

# ---------------- stage C: per-size tuning under the fast path ----------
tim = []
NAMES = [("float16", [1024]), ("float16", [65536]), ("float16", [1048576]),
         ("float16", [4096, 4096]), ("bfloat16", [1024]), ("bfloat16", [65536]),
         ("bfloat16", [1048576]), ("bfloat16", [4096, 4096]),
         ("float32", [1024]), ("float32", [65536]), ("float32", [1048576]),
         ("float32", [4096, 4096])]
for dt, shape in NAMES:
    t = torch.randn(shape, device=DEV, dtype=getattr(torch, dt))
    nn = t.numel()
    e = {"dt": dt, "n": nn, "torch": bench_wall(lambda t=t: torch.neg_(t), 300)}
    exp = t.clone()
    torch.neg_(exp)
    torch.neg_(exp)
    for blk in (1024, 2048, 4096, 8192):
        try:
            g = ((nn + blk - 1) // blk,)
            even = nn % blk == 0
            c = neg_flat.warmup(t, nn, BLOCK=blk, EVEN=even, grid=g)
            c._init_handles()
            ckf = c
            s = tdrv.active.get_current_stream(None)
            if use_hooks:
                a = (g[0], 1, 1, s, ckf.function, ckf.packed_metadata, None, enter, exit_, t, nn, blk, even)
            else:
                a = (g[0], 1, 1, s, ckf.function, ckf.packed_metadata, None, None, None, t, nn, blk, even)
            ckf._run.launch(*a)
            torch.npu.synchronize()
            e[f"flat_{blk}_wall"] = bench_wall(lambda a=a: ckf._run.launch(*a), 200)
        except Exception:
            e[f"flat_{blk}_wall"] = traceback.format_exc()[-120:]
    for blk, npg in ((4096, 40), (4096, 80), (8192, 20), (8192, 40), (8192, 80), (16384, 20)):
        if nn < blk * npg // 2:
            continue
        try:
            ni = (nn + blk * npg - 1) // (blk * npg)
            c = neg_gs.warmup(t, nn, ni, BLOCK=blk, grid=(npg,))
            c._init_handles()
            s = tdrv.active.get_current_stream(None)
            if use_hooks:
                a = (npg, 1, 1, s, c.function, c.packed_metadata, None, enter, exit_, t, nn, ni, blk)
            else:
                a = (npg, 1, 1, s, c.function, c.packed_metadata, None, None, None, t, nn, ni, blk)
            c._run.launch(*a)
            torch.npu.synchronize()
            e[f"gs_{npg}_{blk}_wall"] = bench_wall(lambda a=a, c=c: c._run.launch(*a), 200)
        except Exception:
            e[f"gs_{npg}_{blk}_wall"] = traceback.format_exc()[-120:]
    tim.append(e)
    res["sizes"] = tim
    flush()

flush()
print(json.dumps(res, indent=1)[:9000])
