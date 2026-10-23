"""Dump the signatures/sources of the ascend vector scatter/gather extensions and
re-run the guarded component timings (each measurement isolated so one compile
failure cannot abort the job).
"""
import inspect
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
OUT = {"api": {}, "dev_ms": {}, "errs": {}}


def log(tag, **kw):
    print(json.dumps({"tag": tag, **kw}), flush=True)


import triton.language.extra.cann.extension as al

OUT["module_file"] = getattr(al, "__file__", None)
for name in ("index_put", "scatter_ub_to_out", "gather_out_to_ub", "index_select_simd",
             "copy", "insert_slice", "extract_slice", "flip", "multibuffer", "parallel",
             "sub_vec_num", "sub_vec_id"):
    obj = getattr(al, name, None)
    entry = {"exists": obj is not None}
    if obj is not None:
        try:
            entry["doc"] = (inspect.getdoc(obj) or "")[:1800]
        except Exception as exc:  # noqa: BLE001
            entry["doc_err"] = repr(exc)
        try:
            entry["src"] = inspect.getsource(obj)[:3500]
        except Exception as exc:  # noqa: BLE001
            entry["src_err"] = repr(exc)
        entry["sig"] = str(getattr(obj, "__doc__", ""))[:200]
    OUT["api"][name] = entry
log("api", names={k: bool(v.get("exists")) for k, v in OUT["api"].items()})


# --------------------------------------------------------------- kernels --- #
@triton.jit
def k_atomic(src_ptr, idx_ptr, out_ptr, ncols: tl.constexpr, BLOCK: tl.constexpr,
             MASKED: tl.constexpr):
    row = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    base = row * ncols
    if MASKED:
        m = c < ncols
        idx = tl.load(idx_ptr + base + c, mask=m, other=0).to(tl.int32)
        val = tl.load(src_ptr + base + c, mask=m, other=0).to(tl.float32)
        tl.atomic_add(out_ptr + (base + idx), val, mask=m, sem="relaxed")
    else:
        idx = tl.load(idx_ptr + base + c).to(tl.int32)
        val = tl.load(src_ptr + base + c).to(tl.float32)
        tl.atomic_add(out_ptr + (base + idx), val, sem="relaxed")


@triton.jit
def k_sort_only(idx_ptr, keys_ptr, ncols: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, B)
    m = lane < ncols
    t = tl.load(idx_ptr + row * ncols + lane, mask=m, other=B).to(tl.int32)
    key = al.sort((t * B + lane).to(tl.float32), dim=-1)
    tl.store(keys_ptr + row * B + lane, key)


@triton.jit
def k_hist_only(idx_ptr, out_ptr, ncols: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, B)
    m = lane < ncols
    t = tl.load(idx_ptr + row * ncols + lane, mask=m, other=0).to(tl.int32)
    h = tl.histogram(t, B).to(tl.int32)
    cum = tl.cumsum(h, axis=0)
    tl.store(out_ptr + row * B + lane, cum)


@triton.jit
def k_gather2(pre_ptr, beg_ptr, out_ptr, B: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, B)
    pre = tl.load(pre_ptr + row * B + lane)
    b = tl.load(beg_ptr + row * B + lane)
    g1 = tl.gather(pre, tl.maximum(tl.minimum(b - 1, B - 1), 0), axis=0)
    g2 = tl.gather(pre, tl.maximum(tl.minimum(b + 7, B - 1), 0), axis=0)
    tl.store(out_ptr + row * B + lane, g2 - g1)


@triton.jit
def k_own_f32(inp_ptr, idx_ptr, src_ptr, out_ptr, ncols: tl.constexpr,
              BLOCK: tl.constexpr, CHUNK: tl.constexpr):
    row = tl.program_id(0)
    dst = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    base = row * ncols
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s0 in range(0, ncols, CHUNK):
        so = s0 + tl.arange(0, CHUNK)
        idx = tl.load(idx_ptr + base + so).to(tl.int32)
        val = tl.load(src_ptr + base + so).to(tl.float32)
        acc += tl.sum(tl.where(idx[:, None] == dst[None, :], val[:, None], 0.0), axis=0)
    cur = tl.load(inp_ptr + base + dst).to(tl.float32)
    tl.store(out_ptr + base + dst, (cur + acc).to(out_ptr.dtype.element_ty))


@triton.jit
def k_own_f16(inp_ptr, idx_ptr, src_ptr, out_ptr, ncols: tl.constexpr,
              BLOCK: tl.constexpr, CHUNK: tl.constexpr):
    row = tl.program_id(0)
    dst = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    d16 = dst.to(tl.float16)
    base = row * ncols
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s0 in range(0, ncols, CHUNK):
        so = s0 + tl.arange(0, CHUNK)
        idx = tl.load(idx_ptr + base + so).to(tl.int32).to(tl.float16)
        val = tl.load(src_ptr + base + so).to(tl.float16)
        acc += tl.sum(tl.where(idx[:, None] == d16[None, :], val[:, None], 0.0), axis=0).to(tl.float32)
    cur = tl.load(inp_ptr + base + dst).to(tl.float32)
    tl.store(out_ptr + base + dst, (cur + acc).to(out_ptr.dtype.element_ty))


@triton.jit
def k_bw(idx_ptr, src_ptr, inp_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    i = tl.load(idx_ptr + off, mask=m, other=0)
    v = tl.load(src_ptr + off, mask=m, other=0).to(tl.float32)
    x = tl.load(inp_ptr + off, mask=m, other=0).to(tl.float32)
    tl.store(out_ptr + off, (x + v + i.to(tl.float32)).to(out_ptr.dtype.element_ty), mask=m)


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
        OUT["errs"][name] = repr(exc)[:300]
        log("err", name=name, msg=repr(exc)[:300])
    log("dev", **{name: store.get(name, "ERR")})


def main():
    g = torch.Generator().manual_seed(1729)
    for rows, W in ((1024, 1024), (2048, 2048)):
        B = 1 << (W - 1).bit_length()
        n = rows * W
        dt = torch.float16
        inp = torch.randn((rows, W), generator=g).to(dtype=dt, device=DEV)
        src = torch.randn((rows, W), generator=g).to(dtype=dt, device=DEV)
        idx = torch.randint(W, (rows, W), generator=g).to(DEV)
        out = torch.empty_like(inp)
        buf = torch.empty((rows, W), dtype=torch.float32, device=DEV)
        keys = torch.empty((rows, B), dtype=torch.int32, device=DEV)
        pre = torch.empty((rows, B), dtype=torch.float32, device=DEV)
        cum = torch.empty((rows, B), dtype=torch.int32, device=DEV)
        t = {"bytes_MB": round((n * 8 + n * 2 * 3) / 1e6, 2)}
        tag = f"{rows}x{W}"
        tbench(t, "bw_cal", lambda: k_bw[(triton.cdiv(n, 4096),)](
            idx, src, inp, out, n, BLOCK=4096))
        for blk in (1024, 2048):
            if W % blk:
                continue
            g1 = (rows, W // blk)
            tbench(t, f"atomic_nomask_b{blk}",
                   lambda g1=g1, blk=blk: k_atomic[g1](src, idx, buf, W, BLOCK=blk, MASKED=False))
            tbench(t, f"atomic_mask_b{blk}",
                   lambda g1=g1, blk=blk: k_atomic[g1](src, idx, buf, W, BLOCK=blk, MASKED=True))
        tbench(t, "sort_f32", lambda: k_sort_only[(rows,)](idx, keys, W, B))
        tbench(t, "hist", lambda: k_hist_only[(rows,)](idx, cum, W, B))
        tbench(t, "gather2", lambda: k_gather2[(rows,)](pre, cum, pre, B))
        tbench(t, "own_f32_b128c64", lambda: k_own_f32[(rows, W // 128)](
            inp, idx, src, out, W, BLOCK=128, CHUNK=64))
        tbench(t, "own_f16_b128c64", lambda: k_own_f16[(rows, W // 128)](
            inp, idx, src, out, W, BLOCK=128, CHUNK=64))
        try:
            ref = torch.scatter_add(inp, -1, idx, src)
            k_own_f16[(rows, W // 128)](inp, idx, src, out, W, BLOCK=128, CHUNK=64)
            torch.npu.synchronize()
            d = (out.float() - ref.float()).abs()
            t["own_f16_ok"] = bool((d <= 0.015 + 0.015 * ref.float().abs()).all().item())
            t["own_f16_maxabs"] = round(float(d.max().item()), 6)
        except Exception as exc:  # noqa: BLE001
            t["own_f16_err"] = repr(exc)[:200]
        OUT["dev_ms"][tag] = t
        log("case", case=tag, **t)
        with open(os.path.join(ART, "probe_api.json"), "w") as fh:
            json.dump(OUT, fh, indent=1)
    print("DONE", flush=True)


try:
    main()
except Exception:  # noqa: BLE001
    traceback.print_exc()
    with open(os.path.join(ART, "probe_api.json"), "w") as fh:
        json.dump(OUT, fh, indent=1)
    print("FAILED", flush=True)
