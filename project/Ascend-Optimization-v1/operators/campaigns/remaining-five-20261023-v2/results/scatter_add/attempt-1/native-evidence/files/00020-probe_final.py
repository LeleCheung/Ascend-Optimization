"""Final component probe: isolate k1 cost (sort / gather / cumsum) and validate a
histogram-free combine kernel that finds each destination's sorted slice with a
branchless binary search over global memory (no on-chip gather in the loop).
"""
import json
import os
import traceback

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
DEV = "npu:0"
torch.npu.set_device(0)
OUT = {"dev_ms": {}, "checks": [], "errs": {}}


def log(tag, **kw):
    print(json.dumps({"tag": tag, **kw}), flush=True)


import triton.language.extra.cann.extension as al


@triton.jit
def k1_full(S, I, KEYS, PREFIX, W: tl.constexpr, B: tl.constexpr, DOPREFIX: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, B)
    m = lane < W
    t = tl.load(I + row * W + lane, mask=m, other=B).to(tl.int32)
    key = al.sort((t * B + lane).to(tl.float32), dim=-1).to(tl.int32)
    src_lane = tl.minimum(key % B, W - 1)
    values = tl.load(S + row * W + lane, mask=m, other=0.0).to(tl.float32)
    sv = tl.gather(values, src_lane, axis=0)
    sv = tl.where(key // B < W, sv, 0.0)
    tl.store(KEYS + row * B + lane, key)
    if DOPREFIX:
        tl.store(PREFIX + row * B + lane, tl.cumsum(sv, axis=0))
    else:
        tl.store(PREFIX + row * B + lane, sv)


@triton.jit
def k_sort_gather(S, I, KEYS, PREFIX, W: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, B)
    m = lane < W
    t = tl.load(I + row * W + lane, mask=m, other=B).to(tl.int32)
    key = al.sort((t * B + lane).to(tl.float32), dim=-1).to(tl.int32)
    tl.store(KEYS + row * B + lane, key)


@triton.jit
def k_cumsum_only(PREFIX, src_ptr, B: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, B)
    v = tl.load(src_ptr + row * B + lane)
    tl.store(PREFIX + row * B + lane, tl.cumsum(v, axis=0))


@triton.jit
def k2_search(X, KEYS, PREFIX, O, W: tl.constexpr, B: tl.constexpr, LOG: tl.constexpr):
    row = tl.program_id(0)
    base = row * B
    dst = tl.arange(0, B)
    t1 = dst * B
    lo = tl.zeros([B], tl.int32)
    hi = tl.full([B], B, tl.int32)
    for _ in tl.static_range(LOG + 1):
        active = lo < hi
        mid = tl.minimum((lo + hi) >> 1, B - 1)
        v = tl.load(KEYS + base + mid)
        lo = tl.where(active & (v < t1), mid + 1, lo)
        hi = tl.where(active & (v >= t1), mid, hi)
    nxt = tl.gather(lo, tl.minimum(dst + 1, B - 1), axis=0)
    end = tl.where(dst == W - 1, W, nxt)
    before = tl.load(PREFIX + base + tl.maximum(lo - 1, 0))
    after = tl.load(PREFIX + base + tl.maximum(tl.minimum(end - 1, B - 1), 0))
    delta = tl.where(end > 0, after, 0.0) - tl.where(lo > 0, before, 0.0)
    init = tl.load(X + row * W + dst, mask=dst < W, other=0.0).to(tl.float32)
    tl.store(O + row * W + dst, (init + delta).to(O.dtype.element_ty), mask=dst < W)


def bench(fn, iters=20, warm=3):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    e0 = torch.npu.Event(enable_timing=True)
    e1 = torch.npu.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    e1.synchronize()
    return round(e0.elapsed_time(e1) / iters, 5)


def tbench(store, name, fn):
    try:
        store[name] = bench(fn)
    except Exception as exc:  # noqa: BLE001
        OUT["errs"][name] = repr(exc)[:400]
        log("err", name=name, msg=repr(exc)[:400])
    log("dev", **{name: store.get(name, "ERR")})


def main():
    g = torch.Generator().manual_seed(1729)
    for rows, W in ((1024, 1024), (2048, 2048)):
        B = 1 << (W - 1).bit_length()
        LOG = B.bit_length() - 1
        tag = f"{rows}x{W}"
        inp = torch.randn((rows, W), generator=g).to(dtype=torch.float16, device=DEV)
        src = torch.randn((rows, W), generator=g).to(dtype=torch.float16, device=DEV)
        idx = torch.randint(W, (rows, W), generator=g).to(DEV)
        out = torch.empty_like(inp)
        keys = torch.empty((rows, B), dtype=torch.int32, device=DEV)
        pre = torch.empty((rows, B), dtype=torch.float32, device=DEV)
        t = {}
        tbench(t, "sort_only", lambda: k_sort_gather[(rows,)](src, idx, keys, pre, W, B))
        tbench(t, "k1_nocumsum", lambda: k1_full[(rows,)](src, idx, keys, pre, W, B, DOPREFIX=False))
        tbench(t, "k1_full", lambda: k1_full[(rows,)](src, idx, keys, pre, W, B, DOPREFIX=True))
        tbench(t, "cumsum_only", lambda: k_cumsum_only[(rows,)](pre, pre, B))
        tbench(t, "k2_search", lambda: k2_search[(rows,)](inp, keys, pre, out, W, B, LOG))
        try:
            k1_full[(rows,)](src, idx, keys, pre, W, B, DOPREFIX=True)
            k2_search[(rows,)](inp, keys, pre, out, W, B, LOG)
            e2e = bench(lambda: (k1_full[(rows,)](src, idx, keys, pre, W, B, DOPREFIX=True),
                                 k2_search[(rows,)](inp, keys, pre, out, W, B, LOG)))
            t["e2e"] = e2e
            for dt in (torch.float16, torch.bfloat16, torch.float32):
                i2 = inp.to(dt)
                s2 = src.to(dt)
                o2 = torch.empty_like(i2)
                k1_full[(rows,)](s2, idx, keys, pre, W, B, DOPREFIX=True)
                k2_search[(rows,)](i2, keys, pre, o2, W, B, LOG)
                r = torch.scatter_add(i2, -1, idx, s2)
                torch.npu.synchronize()
                d = (o2.float() - r.float()).abs()
                ok = bool((d <= (0.015 if dt != torch.float32 else 1e-4)
                           + (0.015 if dt != torch.float32 else 1e-4) * r.float().abs()).all().item())
                OUT["checks"].append({"case": f"{tag}-{dt}", "ok": ok,
                                      "max_abs": round(float(d.max().item()), 8)})
                log("check", case=f"{tag}-{dt}", ok=ok, max_abs=round(float(d.max().item()), 8))
        except Exception as exc:  # noqa: BLE001
            OUT["errs"]["e2e"] = repr(exc)[:400]
            log("err", name="e2e", msg=repr(exc)[:400])
        OUT["dev_ms"][tag] = t
        with open(os.path.join(ART, "probe_final.json"), "w") as fh:
            json.dump(OUT, fh, indent=1)
    print("DONE", flush=True)


try:
    main()
except Exception:  # noqa: BLE001
    traceback.print_exc()
    with open(os.path.join(ART, "probe_final.json"), "w") as fh:
        json.dump(OUT, fh, indent=1)
    print("FAILED", flush=True)
