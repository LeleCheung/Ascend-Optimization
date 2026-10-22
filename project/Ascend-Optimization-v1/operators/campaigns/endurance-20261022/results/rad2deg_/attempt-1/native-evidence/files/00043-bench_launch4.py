"""Round 4 diagnosis: cheap current-stream resolution + device-side kernel
variants for the 4096x4096 timing cases. Diagnostic only."""
import json
import os
import time
import traceback
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = Path(os.environ.get("KGS_DEBUG_ARTIFACTS", "."))
ART.mkdir(parents=True, exist_ok=True)
out = {"bench": {}, "errors": {}, "facts": {}}

C = 57.29577951308232


@triton.jit
def _k(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(x_ptr + offs, (x.to(tl.float32) * 57.29577951308232).to(x.dtype), mask=m)


@triton.jit
def _k_nomask(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(x_ptr + offs, (x.to(tl.float32) * 57.29577951308232).to(x.dtype))


@triton.jit
def _k_persist(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    step = tl.num_programs(0) * BLOCK
    for start in range(pid * BLOCK, n, step):
        offs = start + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m)
        tl.store(x_ptr + offs, (x.to(tl.float32) * 57.29577951308232).to(x.dtype), mask=m)


def bench(fn, iters=40, warm=6):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def bench_small(fn, iters=300, warm=30):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


# ---- stream resolution candidates --------------------------------------
try:
    import triton.backends.ascend.driver as adrv
    out["facts"]["get_backend_func_src"] = repr(adrv.get_backend_func)[:600]
    drv = triton.runtime.driver.active
    f = drv.get_current_stream
    out["bench"]["drv_bound_us"] = bench_small(lambda: f(0))
    try:
        out["bench"]["get_backend_func_us"] = bench_small(lambda: adrv.get_backend_func("get_current_stream", 0))
    except Exception:
        out["errors"]["gbf"] = traceback.format_exc()[-300:]
    # underlying torch_npu helpers
    for name, fn in [
        ("torch_C_npu_getCurStream", lambda: torch._C._npu_getCurrentStream()),
        ("torch_npu_current_stream", lambda: torch.npu.current_stream()._as_parameter_),
    ]:
        try:
            v = fn()
            out["facts"][name] = str(v)[:80]
            out["bench"][name + "_us"] = bench_small(fn)
        except Exception:
            out["errors"][name] = traceback.format_exc()[-200:]
except Exception:
    out["errors"]["stream"] = traceback.format_exc()[-500:]

st = triton.runtime.driver.active.get_current_stream(0)

# ---- large-case kernel variants ---------------------------------------
B = torch.randn(4096, 4096, dtype=torch.float16, device="npu:0")
nb = B.numel()
out["facts"]["n_big"] = nb


def run_variant(kern, grid, blk, expect_ok, **opts):
    try:
        ck = kern.warmup(B, nb, BLOCK=blk, grid=(grid, 1, 1), **opts)
        ref = None
        B.copy_(torch.randint(-1000, 1000, (4096, 4096), device="npu:0").half() / 1000.0)
        ref = torch.rad2deg(B.float()).half()
        ck.run.launch(grid, 1, 1, st, ck.function, ck.packed_metadata, None, None, None,
                      B, nb, blk)
        torch.npu.synchronize()
        ok = bool(torch.equal(B, ref))
        t = bench(lambda: ck.run.launch(grid, 1, 1, st, ck.function, ck.packed_metadata, None, None, None,
                                        B, nb, blk))
        return {"us": t, "correct": ok}
    except Exception:
        return {"err": traceback.format_exc().strip().splitlines()[-1][:200]}


variants = {}
for blk in (16384, 32768):
    g = (nb + blk - 1) // blk
    variants[f"mask_blk{blk}"] = run_variant(_k, g, blk, True)
for blk in (16384, 32768):
    g = (nb + blk - 1) // blk
    variants[f"nomask_blk{blk}"] = run_variant(_k_nomask, g, blk, True)
for nw in (1, 4, 8, 16, 32):
    g = (nb + 32767) // 32768
    variants[f"mask_blk32768_w{nw}"] = run_variant(_k, g, 32768, True, num_warps=nw)
for nprog in (40, 80, 160):
    g = nprog
    variants[f"persist_p{nprog}_blk32768"] = run_variant(_k_persist, g, 32768, True)
for blk, mb in ((65536, False),):
    g = (nb + blk - 1) // blk
    variants[f"mask_blk{blk}_mb{mb}"] = run_variant(_k, g, blk, True, multibuffer=mb)

variants["ref_tight"] = {"us": bench(lambda: torch.rad2deg_(B))}
out["bench"]["big_variants"] = variants

# ---- small-case launch floor with int ptr, int stream ------------------
A = torch.randn(1024, dtype=torch.float16, device="npu:0")
na = A.numel()
g = (na + 2047) // 2048
try:
    ck = _k.warmup(A, na, BLOCK=2048, grid=(1, 1, 1))
    l, f, m = ck.run.launch, ck.function, ck.packed_metadata
    ap = A.data_ptr()
    out["facts"]["intptr_correct"] = None
    A.copy_(torch.randn(1024, dtype=torch.float16, device="npu:0"))
    ref = torch.rad2deg(A.float()).half()
    l(1, 1, 1, st, f, m, None, None, None, A.data_ptr(), na, 2048)
    torch.npu.synchronize()
    out["facts"]["intptr_correct"] = bool(torch.equal(A, ref))
    out["bench"]["floor_launch_intptr_us"] = bench_small(
        lambda: l(1, 1, 1, st, f, m, None, None, None, ap, na, 2048))
    out["bench"]["floor_plus_stream_us"] = bench_small(
        lambda: l(1, 1, 1, triton.runtime.driver.active.get_current_stream(0), f, m, None, None, None, ap, na, 2048))
    out["bench"]["ref_small_tight"] = bench_small(lambda: torch.rad2deg_(A))
except Exception:
    out["errors"]["floor"] = traceback.format_exc()[-600:]

# real end-to-end candidate shape (dict lookup + numel + contiguity + launch)
try:
    cache = {}
    OBJ = triton.runtime.driver.active
    gsc = OBJ.get_current_stream

    def cand(A):
        n = A.numel()
        e = cache.get(A.dtype)
        if e is None:
            blk = 2048
            gg = (n + blk - 1) // blk
            ck = _k.warmup(A, n, BLOCK=blk, grid=(gg, 1, 1))
            e = cache[A.dtype] = (ck.run.launch, ck.function, ck.packed_metadata, blk)
        l, f, m, blk = e
        l((n + blk - 1) // blk, 1, 1, gsc(0), f, m, None, None, None, A.data_ptr(), n, blk)
        return A

    cand(A)
    torch.npu.synchronize()
    out["bench"]["cand_end_to_end_us"] = bench_small(lambda: cand(A))
except Exception:
    out["errors"]["cand"] = traceback.format_exc()[-600:]

(ART / "bench4.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
print(json.dumps(out["bench"], indent=2, default=str)[:5000])
print("FACTS", json.dumps(out["facts"], default=str)[:1500])
print("ERRORS", list(out["errors"].keys()))
