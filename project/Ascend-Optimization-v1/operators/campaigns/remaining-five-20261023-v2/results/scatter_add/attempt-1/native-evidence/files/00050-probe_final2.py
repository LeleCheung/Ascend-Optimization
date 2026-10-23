"""Diagnose the k2_search compile failure (full error) and time a k1 that avoids
integer div/mod, testing three combine variants so at least one compiles.
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
OUT = {"dev_ms": {}, "errs": {}, "checks": []}


def log(tag, **kw):
    print(json.dumps({"tag": tag, **kw}), flush=True)


import triton.language.extra.cann.extension as al


@triton.jit
def k1_shift(S, I, KEYS, PREFIX, W: tl.constexpr, B: tl.constexpr, LOG: tl.constexpr,
             DOPREFIX: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, B)
    m = lane < W
    t = tl.load(I + row * W + lane, mask=m, other=B).to(tl.int32)
    key = al.sort((t * B + lane).to(tl.float32), dim=-1).to(tl.int32)
    src_lane = tl.minimum(key & (B - 1), W - 1)
    values = tl.load(S + row * W + lane, mask=m, other=0.0).to(tl.float32)
    sv = tl.gather(values, src_lane, axis=0)
    sv = tl.where((key >> LOG) < W, sv, 0.0)
    tl.store(KEYS + row * B + lane, key)
    if DOPREFIX:
        tl.store(PREFIX + row * B + lane, tl.cumsum(sv, axis=0))
    else:
        tl.store(PREFIX + row * B + lane, sv)


@triton.jit
def k2_v1(X, KEYS, PREFIX, O, W: tl.constexpr, B: tl.constexpr, LOG: tl.constexpr):
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


@triton.jit
def k2_v2(X, KEYS, PREFIX, O, W: tl.constexpr, B: tl.constexpr, LOG: tl.constexpr):
    """nested tl.where, no bitwise and on predicates."""
    row = tl.program_id(0)
    base = row * B
    dst = tl.arange(0, B)
    t1 = dst * B
    lo = tl.zeros([B], tl.int32)
    hi = tl.full([B], B, tl.int32)
    for _ in tl.static_range(LOG + 1):
        active = lo < hi
        mid = tl.minimum((lo + hi) >> 1, B - 1)
        less = tl.load(KEYS + base + mid) < t1
        lo = tl.where(active, tl.where(less, mid + 1, lo), lo)
        hi = tl.where(active, tl.where(less, hi, mid), hi)
    nxt = tl.gather(lo, tl.minimum(dst + 1, B - 1), axis=0)
    end = tl.where(dst == W - 1, W, nxt)
    before = tl.load(PREFIX + base + tl.maximum(lo - 1, 0))
    after = tl.load(PREFIX + base + tl.maximum(tl.minimum(end - 1, B - 1), 0))
    delta = tl.where(end > 0, after, 0.0) - tl.where(lo > 0, before, 0.0)
    init = tl.load(X + row * W + dst, mask=dst < W, other=0.0).to(tl.float32)
    tl.store(O + row * W + dst, (init + delta).to(O.dtype.element_ty), mask=dst < W)


@triton.jit
def k2_v3(X, KEYS, PREFIX, O, W: tl.constexpr, B: tl.constexpr, LOG: tl.constexpr):
    """search only the second boundary by interpolation-free plain bisection on
    int64 indices held as fp32-free int32; identical math, no predicate and-op."""
    row = tl.program_id(0)
    base = row * B
    dst = tl.arange(0, B)
    t1 = dst * B
    lo = tl.zeros([B], tl.int32)
    hi = tl.full([B], B, tl.int32)
    for _ in tl.static_range(LOG + 1):
        mid = tl.minimum((lo + hi) >> 1, B - 1)
        v = tl.load(KEYS + base + mid)
        go = v < t1
        go2 = go & (lo < hi)
        lo = tl.where(go2, mid + 1, lo)
        hi = tl.where(go2, hi, tl.minimum(mid, hi))
    end = tl.minimum(lo + 1, B)
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
        OUT["errs"][name] = repr(exc)[:6000]
        log("err", name=name, msg=repr(exc)[:1200])
    log("dev", **{name: store.get(name, "ERR")})


def main():
    variants = {"v1": k2_v1, "v2": k2_v2, "v3": k2_v3}
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
        tbench(t, "k1_shift_nopre",
               lambda: k1_shift[(rows,)](src, idx, keys, pre, W, B, LOG, DOPREFIX=False))
        tbench(t, "k1_shift_full",
               lambda: k1_shift[(rows,)](src, idx, keys, pre, W, B, LOG, DOPREFIX=True))
        for name, kern in variants.items():
            tbench(t, f"k2_{name}", lambda kern=kern: kern[(rows,)](
                inp, keys, pre, out, W, B, LOG))
        # end-to-end + numerics for whichever variant compiled
        for name, kern in variants.items():
            try:
                k1_shift[(rows,)](src, idx, keys, pre, W, B, LOG, DOPREFIX=True)
                kern[(rows,)](inp, keys, pre, out, W, B, LOG)
                torch.npu.synchronize()
                t[f"e2e_{name}"] = bench(lambda: (k1_shift[(rows,)](
                    src, idx, keys, pre, W, B, LOG, DOPREFIX=True),
                    kern[(rows,)](inp, keys, pre, out, W, B, LOG)))
                for dt in (torch.float16, torch.bfloat16, torch.float32):
                    i2, s2 = inp.to(dt), src.to(dt)
                    o2 = torch.empty_like(i2)
                    k1_shift[(rows,)](s2, idx, keys, pre, W, B, LOG, DOPREFIX=True)
                    kern[(rows,)](i2, keys, pre, o2, W, B, LOG)
                    r = torch.scatter_add(i2, -1, idx, s2)
                    torch.npu.synchronize()
                    d = (o2.float() - r.float()).abs()
                    tol = 1e-4 if dt == torch.float32 else 0.015
                    OUT["checks"].append({"case": f"{name}-{tag}-{dt}",
                                          "ok": bool((d <= tol + tol * r.float().abs()).all().item()),
                                          "max_abs": round(float(d.max().item()), 8)})
                    log("check", **OUT["checks"][-1])
            except Exception as exc:  # noqa: BLE001
                OUT["errs"][f"e2e_{name}"] = repr(exc)[:1500]
                log("err", name=f"e2e_{name}", msg=repr(exc)[:400])
        OUT["dev_ms"][tag] = t
        log("case", case=tag, **t)
        with open(os.path.join(ART, "probe_final2.json"), "w") as fh:
            json.dump(OUT, fh, indent=1)
    print("DONE", flush=True)


try:
    main()
except Exception:  # noqa: BLE001
    traceback.print_exc()
    with open(os.path.join(ART, "probe_final2.json"), "w") as fh:
        json.dump(OUT, fh, indent=1)
    print("FAILED", flush=True)
