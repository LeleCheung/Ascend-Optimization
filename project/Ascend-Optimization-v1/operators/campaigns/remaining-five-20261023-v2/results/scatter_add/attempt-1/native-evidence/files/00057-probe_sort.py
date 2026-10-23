"""Target probe: sorted-segment scatter_add vs torch reference.

Hypothesis under test: the O(w^2) owner kernel and the per-lane atomic path can
both be replaced by a sort-based segmented reduction:

  k1: per row, sort (target*B + lane) keys, reorder src by source lane, store
      keys + inclusive cumsum of the reordered values.
  k2: per row, derive each destination's [begin, end) slice of the sorted keys
      (histogram+cumsum, or a binary search fallback) and emit
      out[dst] = inp[dst] + prefix[end-1] - prefix[begin-1] contiguously.

Because keys sort by (target, lane) the segment sum is accumulated in ascending
column order, matching the reference's column-ordered fp32 accumulation.
"""
import json
import os
import time
import traceback

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
DEV = "npu:0"
torch.npu.set_device(0)
ENV = {}


def log(tag, **kw):
    print(json.dumps({"tag": tag, **kw}), flush=True)


try:
    import triton.language.extra.cann.extension as al
    ENV["al_has_sort"] = hasattr(al, "sort")
except Exception as exc:  # noqa: BLE001
    ENV["al_import_error"] = repr(exc)
ENV["has_histogram"] = hasattr(tl, "histogram")
log("env", **ENV)


@triton.jit
def k1_sort(S, I, KEYS, PREFIX, W: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, B)
    m = lane < W
    target = tl.load(I + row * W + lane, mask=m, other=B).to(tl.int32)
    encoded = target * B + lane
    key = al.sort(encoded.to(tl.float32), dim=-1).to(tl.int32)
    src_lane = tl.minimum(key % B, W - 1)
    values = tl.load(S + row * W + lane, mask=m, other=0.0).to(tl.float32)
    sv = tl.gather(values, src_lane, axis=0)
    sv = tl.where(key // B < W, sv, 0.0)
    prefix = tl.cumsum(sv, axis=0)
    tl.store(KEYS + row * B + lane, key)
    tl.store(PREFIX + row * B + lane, prefix)


@triton.jit
def k2_hist(X, I, KEYS, PREFIX, O, W: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, B)
    dst = tl.arange(0, B)
    m = lane < W
    target = tl.load(I + row * W + lane, mask=m, other=0).to(tl.int32)
    hist = tl.histogram(target, B).to(tl.int32) - tl.where(dst == 0, B - W, 0)
    cum = tl.cumsum(hist, axis=0)
    begin = cum - hist
    prefix = tl.load(PREFIX + row * B + dst)
    before = tl.gather(prefix, tl.maximum(tl.minimum(begin - 1, B - 1), 0), axis=0)
    after = tl.gather(prefix, tl.maximum(tl.minimum(cum - 1, B - 1), 0), axis=0)
    delta = tl.where(begin > 0, after - before, tl.where(cum > 0, after, 0.0))
    init = tl.load(X + row * W + dst, mask=dst < W, other=0.0).to(tl.float32)
    tl.store(O + row * W + dst, (init + delta).to(O.dtype.element_ty), mask=dst < W)


@triton.jit
def k2_search(X, KEYS, PREFIX, O, W: tl.constexpr, B: tl.constexpr, LOG: tl.constexpr):
    row = tl.program_id(0)
    dst = tl.arange(0, B)
    keys = tl.load(KEYS + row * B + dst)
    prefix = tl.load(PREFIX + row * B + dst)
    t1 = dst * B
    t2 = (dst + 1) * B
    lo = tl.zeros([B], tl.int32)
    hi = tl.full([B], B, tl.int32)
    lo2 = tl.zeros([B], tl.int32)
    hi2 = tl.full([B], B, tl.int32)
    for _ in tl.static_range(LOG + 1):
        mid = (lo + hi) // 2
        v = tl.gather(keys, tl.minimum(mid, B - 1), axis=0)
        less = (mid < B) & (v < t1)
        lo = tl.where(less, mid + 1, lo)
        hi = tl.where(less, hi, mid)
        mid2 = (lo2 + hi2) // 2
        v2 = tl.gather(keys, tl.minimum(mid2, B - 1), axis=0)
        less2 = (mid2 < B) & (v2 < t2)
        lo2 = tl.where(less2, mid2 + 1, lo2)
        hi2 = tl.where(less2, hi2, mid2)
    begin = lo
    end = lo2
    before = tl.gather(prefix, tl.maximum(tl.minimum(begin - 1, B - 1), 0), axis=0)
    after = tl.gather(prefix, tl.maximum(tl.minimum(end - 1, B - 1), 0), axis=0)
    delta = tl.where(end > 0, after, 0.0) - tl.where(begin > 0, before, 0.0)
    init = tl.load(X + row * W + dst, mask=dst < W, other=0.0).to(tl.float32)
    tl.store(O + row * W + dst, (init + delta).to(O.dtype.element_ty), mask=dst < W)


def run_hist(inp, index, src, W, rows, B):
    keys = torch.empty((rows, B), dtype=torch.int32, device=inp.device)
    prefix = torch.empty((rows, B), dtype=torch.float32, device=inp.device)
    out = torch.empty_like(inp)
    k1_sort[(rows,)](src, index, keys, prefix, W, B)
    k2_hist[(rows,)](inp, index, keys, prefix, out, W, B)
    return out


def run_search(inp, index, src, W, rows, B):
    keys = torch.empty((rows, B), dtype=torch.int32, device=inp.device)
    prefix = torch.empty((rows, B), dtype=torch.float32, device=inp.device)
    out = torch.empty_like(inp)
    k1_sort[(rows,)](src, index, keys, prefix, W, B)
    k2_search[(rows,)](inp, keys, prefix, out, W, B, B.bit_length() - 1)
    return out


def bench_wall(fn, iters=20):
    fn()
    fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def make_case(rows, W, dtype, seed=1729):
    g = torch.Generator().manual_seed(seed)
    shape = (rows, W)
    inp = torch.randn(shape, generator=g).to(dtype=dtype, device=DEV)
    src = torch.randn(shape, generator=g).to(dtype=dtype, device=DEV)
    idx = torch.randint(W, shape, generator=g).to(DEV)
    return inp, idx, src


def check(name, fn, inp, idx, src):
    ref = torch.scatter_add(inp, -1, idx, src)
    got = fn(inp, idx, src)
    torch.npu.synchronize()
    if got.dtype != ref.dtype or got.shape != ref.shape:
        return {"case": name, "ok": False, "reason": "dtype/shape"}
    d = (got.float() - ref.float()).abs()
    tol = 1.5e-2 if inp.dtype in (torch.float16, torch.bfloat16) else 1e-4
    return {"case": name, "ok": bool((d <= tol).all().item()),
            "max_abs": round(float(d.max().item()), 9), "tol": tol}


def main():
    res = {"env": ENV, "checks": [], "timing": {}}
    variants = [("hist", run_hist), ("search", run_search)]
    # ---------- correctness ----------
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for rows, W in ((35, 9), (128, 128), (256, 1024), (512, 2048)):
            B = 1 << (W - 1).bit_length()
            inp, idx, src = make_case(rows, W, dtype)
            for tag, fn in variants:
                if tag == "search" and W > 1024:
                    continue
                try:
                    r = check(f"{tag}-{dtype}-{rows}x{W}",
                              lambda a, b, c: fn(a, b, c, W, rows, B), inp, idx, src)
                except Exception as exc:  # noqa: BLE001
                    r = {"case": f"{tag}-{dtype}-{rows}x{W}", "ok": False,
                         "error": repr(exc)[:500]}
                res["checks"].append(r)
                log("check", **r)
    # ---------- timing on frozen shapes ----------
    for dtype_name in ("float16", "bfloat16", "float32"):
        dtype = getattr(torch, dtype_name)
        for rows, W in ((128, 128), (1024, 1024), (2048, 2048)):
            B = 1 << (W - 1).bit_length()
            inp, idx, src = make_case(rows, W, dtype, seed=1735)
            e = {}
            try:
                e["ref_wall_ms"] = round(bench_wall(
                    lambda: torch.scatter_add(inp, -1, idx, src)), 5)
            except Exception as exc:  # noqa: BLE001
                e["ref_wall_ms"] = repr(exc)[:200]
            for tag, fn in variants:
                key = f"{tag}_wall_ms"
                try:
                    e[key] = round(bench_wall(lambda: fn(inp, idx, src, W, rows, B)), 5)
                except Exception as exc:  # noqa: BLE001
                    e[key] = repr(exc)[:250]
            res["timing"][f"{dtype_name}-{rows}x{W}"] = e
            log("timing", case=f"{dtype_name}-{rows}x{W}", **e)

    with open(os.path.join(ART, "probe_sort.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("DONE", flush=True)


try:
    main()
except Exception:  # noqa: BLE001
    traceback.print_exc()
    print("FAILED", flush=True)
