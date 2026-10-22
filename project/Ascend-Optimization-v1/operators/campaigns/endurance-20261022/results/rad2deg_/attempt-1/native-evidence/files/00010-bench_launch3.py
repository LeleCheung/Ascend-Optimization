"""Round 3 diagnosis: exact NPU launcher ABI, cheapest correct call form, and
device-side throughput at scale. Diagnostic only."""
import json
import os
import re
import time
import traceback
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = Path(os.environ.get("KGS_DEBUG_ARTIFACTS", "."))
ART.mkdir(parents=True, exist_ok=True)
TR = Path("/usr/local/python3.11.15/lib/python3.11/site-packages/triton")
out = {"bench": {}, "errors": {}, "src": {}}


@triton.jit
def _k(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.store(x_ptr + offs, (x.to(tl.float32) * 57.29577951308232).to(x.dtype), mask=m)


def bench(fn, iters=300, warm=30):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def grab(path, names, tag):
    try:
        txt = Path(path).read_text()
        for pat in names:
            m = re.search(rf"^[ \t]*def {pat}\(.*?(?=^[ \t]*def |^class |\Z)", txt, re.S | re.M)
            if m:
                out["src"][f"{tag}.{pat}"] = m.group(0)[:9000]
            else:
                out["src"][f"{tag}.{pat}"] = "NOT FOUND"
    except Exception:
        out["errors"][f"src.{tag}"] = traceback.format_exc()[-300:]


grab(TR / "backends/ascend/driver.py", ["get_current_stream", "make_launcher", "make_npu_launcher_stub"], "driver")
grab(TR / "spec/ascend/runtime/jit.py", ["run", "getitem"], "jit")

n = 1024
A = torch.randn(n, dtype=torch.float16, device="npu:0")
grid = (n + 8191) // 8192

compiled = _k.warmup(A, n, BLOCK=8192, grid=(grid, 1, 1))
raw = compiled.run
launch_c = raw.launch

# generated launcher stub source (contains the argument parse format)
try:
    hits = []
    cache = Path.home() / ".triton" / "cache"
    for p in cache.rglob("*"):
        if p.is_file() and p.suffix in (".c", ".cpp", ".h", ".py", ".txt") and p.stat().st_size < 400000:
            try:
                t = p.read_text(errors="ignore")
            except Exception:
                continue
            if "PyArg_ParseTuple" in t or "parse_tuple" in t:
                hits.append({"path": str(p), "text": t[:5000]})
    out["src"]["stub_candidates"] = hits[:3]
except Exception:
    out["errors"]["stub"] = traceback.format_exc()[-400:]

st = triton.runtime.driver.active.get_current_stream(0)
fn = compiled.function
meta = compiled.packed_metadata

# ---- does the 12-arg "trailing None" form actually compute correctly? ----
try:
    C = torch.randn(4096, dtype=torch.float16, device="npu:0")
    C0 = C.clone()
    nc = C.numel()
    g2 = (nc + 8191) // 8192
    ck = _k.warmup(C, nc, BLOCK=8192, grid=(g2, 1, 1))
    ck.run(g2, 1, 1, st, ck.function, ck.packed_metadata, None, None, None, C, nc, None)
    torch.npu.synchronize()
    ref = torch.rad2deg(C0.float()).half()
    out["facts_raw_correct_trailing"] = bool(torch.equal(C, ref))
    out["facts_raw_max_err"] = float((C.float() - ref.float()).abs().max())
except Exception:
    out["errors"]["raw_correct"] = traceback.format_exc()[-600:]

# try 11-arg form (no trailing) to confirm it is rejected
try:
    ck.run(g2, 1, 1, st, ck.function, ck.packed_metadata, None, None, None, C, nc)
    out["facts_raw_11arg"] = "accepted"
except Exception as e:
    out["facts_raw_11arg"] = repr(e)[:200]

# ---- launch-form microbenchmarks ---------------------------------------
out["bench"]["jit_full_us"] = bench(lambda: _k[(grid,)](A, n, BLOCK=8192))
out["bench"]["npu_launcher_call_us"] = bench(lambda: raw(grid, 1, 1, st, fn, meta, None, None, None, A, n, None))
out["bench"]["c_launch_direct_us"] = bench(lambda: launch_c(grid, 1, 1, st, fn, meta, None, None, None, A, n, None))
try:
    aptr = A.data_ptr()
    out["bench"]["c_launch_intptr_us"] = bench(lambda: launch_c(grid, 1, 1, st, fn, meta, None, None, None, aptr, n, None))
except Exception:
    out["errors"]["intptr"] = traceback.format_exc()[-300:]
out["bench"]["ref_tight_us"] = bench(lambda: torch.rad2deg_(A))
out["bench"]["noop_us"] = bench(lambda: None)

# ---- realistic end-to-end prototype (cached launcher + stream query) ---
_cache = {}


def proto(A):
    nn = A.numel()
    if nn == 0:
        return A
    e = _cache.get(A.dtype)
    if e is None:
        nn2 = nn
        g = (nn2 + 8191) // 8192
        c = _k.warmup(A, nn2, BLOCK=8192, grid=(g, 1, 1))
        e = _cache[A.dtype] = (c.run.launch, c.function, c.packed_metadata, g)
    l, f, m, g = e
    s = triton.runtime.driver.active.get_current_stream(0)
    l(g, 1, 1, s, f, m, None, None, None, A, nn, None)
    return A


try:
    proto(A)
    torch.npu.synchronize()
    out["bench"]["proto_us"] = bench(lambda: proto(A))
except Exception:
    out["errors"]["proto"] = traceback.format_exc()[-500:]

# ---- device throughput at scale (host overhead ~19us, so >19us is device) ----
def per_iter(fn, iters=30, warm=5):
    return bench(fn, iters=iters, warm=warm)


try:
    B = torch.randn(4096, 4096, dtype=torch.float16, device="npu:0")
    nb = B.numel()
    dev = {}
    for blk in (8192, 16384, 32768):
        g = (nb + blk - 1) // blk
        ck = _k.warmup(B, nb, BLOCK=blk, grid=(g, 1, 1))
        dev[f"raw_blk{blk}"] = per_iter(
            lambda ck=ck, g=g: ck.run.launch(g, 1, 1, st, ck.function, ck.packed_metadata, None, None, None, B, nb, None)
        )
    dev["ref_16M_tight"] = per_iter(lambda: torch.rad2deg_(B))
    out["bench"]["big16M_us"] = dev
except Exception:
    out["errors"]["big"] = traceback.format_exc()[-600:]

# fp32 16M reference
try:
    F = torch.randn(4096, 4096, dtype=torch.float32, device="npu:0")
    nf = F.numel()
    d = {}
    for blk in (8192, 16384, 32768):
        g = (nf + blk - 1) // blk
        ck = _k.warmup(F, nf, BLOCK=blk, grid=(g, 1, 1))
        d[f"raw_blk{blk}"] = per_iter(
            lambda ck=ck, g=g: ck.run.launch(g, 1, 1, st, ck.function, ck.packed_metadata, None, None, None, F, nf, None)
        )
    d["ref_16M_tight"] = per_iter(lambda: torch.rad2deg_(F))
    out["bench"]["big16M_fp32_us"] = d
except Exception:
    out["errors"]["big32"] = traceback.format_exc()[-600:]

(ART / "bench3.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
print(json.dumps(out["bench"], indent=2, default=str)[:4000])
print("FACTS", json.dumps({k: v for k, v in out.items() if k.startswith("facts")}, default=str))
print("ERRORS", list(out["errors"].keys()))
