"""Round 5 diagnosis: direct raw-stream binding, launcher __call__ vs .launch,
and pipelined/persistent big-tensor variants. Diagnostic only."""
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


@triton.jit
def _k(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(x_ptr + offs, (x.to(tl.float32) * 57.29577951308232).to(x.dtype), mask=m)


@triton.jit
def _k_x2(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * (2 * BLOCK)
    for i in tl.static_range(2):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m)
        tl.store(x_ptr + offs, (x.to(tl.float32) * 57.29577951308232).to(x.dtype), mask=m)


@triton.jit
def _k_persist(x_ptr, n, niter, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    for k in tl.range(0, niter):
        offs = (pid + k * npg) * BLOCK + tl.arange(0, BLOCK)
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


# ---- direct raw stream binding -----------------------------------------
drv_stream = triton.runtime.driver.active.get_current_stream
raw_stream = None
try:
    has_nw = hasattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait")
    out["facts"]["has_NoWait"] = has_nw
    if has_nw:
        from torch_npu._C import _npu_getCurrentRawStreamNoWait as raw_stream
    else:
        from torch_npu._C import _npu_getCurrentRawStream as raw_stream
    v0 = raw_stream(0)
    v1 = drv_stream(0)
    out["facts"]["raw_stream_value"] = str(v0)
    out["facts"]["drv_stream_value"] = str(v1)
    out["facts"]["stream_equal"] = int(v0) == int(v1)
    out["bench"]["raw_stream_us"] = bench_small(lambda: raw_stream(0))
    out["bench"]["drv_stream_us"] = bench_small(lambda: drv_stream(0))
except Exception:
    out["errors"]["raw_stream"] = traceback.format_exc()[-500:]

st = drv_stream(0)

# ---- launcher call forms with int pointer ------------------------------
A = torch.randn(1024, dtype=torch.float16, device="npu:0")
na = A.numel()
ck = _k.warmup(A, na, BLOCK=2048, grid=(1, 1, 1))
ap = A.data_ptr()
try:
    out["bench"]["launch_c_intptr_us"] = bench_small(
        lambda: ck.run.launch(1, 1, 1, st, ck.function, ck.packed_metadata, None, None, None, ap, na, 2048))
except Exception:
    out["errors"]["c"] = traceback.format_exc()[-300:]
try:
    out["bench"]["runner_call_intptr_us"] = bench_small(
        lambda: ck.run(1, 1, 1, st, ck.function, ck.packed_metadata, None, None, None, ap, na, 2048))
except Exception:
    out["errors"]["r"] = traceback.format_exc()[-300:]

# full candidate-shaped fast path with the raw stream binding
try:
    if raw_stream is not None:
        def cand(A):
            n = A.numel()
            ck.run.launch((n + 2047) // 2048, 1, 1, raw_stream(0), ck.function, ck.packed_metadata,
                          None, None, None, A.data_ptr(), n, 2048)
            return A

        cand(A)
        torch.npu.synchronize()
        out["bench"]["cand_raw_stream_us"] = bench_small(lambda: cand(A))
        out["bench"]["ref_small_tight"] = bench_small(lambda: torch.rad2deg_(A))
except Exception:
    out["errors"]["cand"] = traceback.format_exc()[-500:]

# ---- big-tensor variants ----------------------------------------------
B = torch.randn(4096, 4096, dtype=torch.float16, device="npu:0")
nb = B.numel()


def var(kern, grid, blk, extra=(), nparams=3):
    try:
        ck = kern.warmup(B, nb, *extra, BLOCK=blk, grid=(grid, 1, 1))
        B.copy_(torch.randint(-1000, 1000, (4096, 4096), device="npu:0").half() / 1000.0)
        ref = torch.rad2deg(B.float()).half()
        ck.run.launch(grid, 1, 1, st, ck.function, ck.packed_metadata, None, None, None,
                      B.data_ptr(), nb, *extra, blk)
        torch.npu.synchronize()
        ok = bool(torch.equal(B, ref))
        t = bench(lambda: ck.run.launch(grid, 1, 1, st, ck.function, ck.packed_metadata, None, None, None,
                                        B.data_ptr(), nb, *extra, blk))
        return {"us": t, "ok": ok}
    except Exception:
        return {"err": traceback.format_exc().strip().splitlines()[-1][:160]}


big = {
    "base_blk32768": var(_k, (nb + 32767) // 32768, 32768),
    "x2_blk32768": var(_k_x2, (nb + 65535) // 65536, 32768),
    "x2_blk16384": var(_k_x2, (nb + 32767) // 32768, 16384),
}
for nprog in (40, 80):
    niter = (nb + nprog * 32768 - 1) // (nprog * 32768)
    big[f"persist_p{nprog}_n{niter}"] = var(_k_persist, nprog, 32768, extra=(niter,), nparams=4)
big["ref_tight"] = {"us": bench(lambda: torch.rad2deg_(B))}
out["bench"]["big"] = big

(ART / "bench5.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
print(json.dumps(out["bench"], indent=2, default=str)[:4000])
print("FACTS", json.dumps(out["facts"], default=str)[:800])
print("ERRORS", list(out["errors"].keys()))
