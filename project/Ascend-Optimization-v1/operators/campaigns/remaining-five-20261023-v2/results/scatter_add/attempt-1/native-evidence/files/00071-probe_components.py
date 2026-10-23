"""Component probe: where does scatter_add time actually go on this backend?

Measures, with device events, the isolated cost of each candidate ingredient on
the frozen 1024x1024 / 2048x2048 fp16 shapes:
  * atomic_add variants (masked vs unmasked, block size, fp32 vs f16 target)
  * al.sort alone (int32 vs fp32 keys)
  * tl.histogram+cumsum alone
  * tl.gather alone
  * owner (compare-fold) variants: fp32 compare vs fp16 compare, tile shapes
  * contiguous bandwidth calibration
and introspects the ascend Triton extension namespace plus installed FlagGems
_ascend ops for any vector scatter / index-put primitive.
"""
import glob
import json
import os
import re
import time
import traceback

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
DEV = "npu:0"
torch.npu.set_device(0)


def log(tag, **kw):
    print(json.dumps({"tag": tag, **kw}), flush=True)


# ------------------------------------------------------------------ env ---- #
env = {"triton": triton.__version__}
try:
    import triton.language.extra.cann.extension as al
    env["al"] = sorted(n for n in dir(al) if not n.startswith("_"))
except Exception as exc:  # noqa: BLE001
    env["al_error"] = repr(exc)
env["tl_has"] = {n: hasattr(tl, n) for n in
                 ("histogram", "gather", "sort", "scatter", "atomic_add")}
hits = {}
for pat in ("/usr/local/lib/python*/dist-packages/flag_gems/runtime/backend/_ascend/ops/*.py",
            "/usr/local/lib/python*/site-packages/flag_gems/runtime/backend/_ascend/ops/*.py",
            "/usr/lib/python*/site-packages/flag_gems/runtime/backend/_ascend/ops/*.py",
            os.path.expanduser("~") + "/**/flag_gems/runtime/backend/_ascend/ops/*.py"):
    for f in glob.glob(pat, recursive=True):
        try:
            txt = open(f).read()
        except Exception:  # noqa: BLE001
            continue
        for m in set(re.findall(r"\b(?:al|extension)\.([a-z_0-9]+)\s*\(", txt)):
            hits.setdefault(m, []).append(os.path.basename(f))
env["al_ext_calls_in_flaggems"] = {k: v[:4] for k, v in sorted(hits.items())}
log("env", **env)


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
def k_sort_i32(idx_ptr, keys_ptr, ncols: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, B)
    m = lane < ncols
    t = tl.load(idx_ptr + row * ncols + lane, mask=m, other=B).to(tl.int32)
    key = al.sort(t * B + lane, dim=-1)
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
    """compare/select in fp16 (exact for indices <= 2048), accumulate fp32."""
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


def main():
    res = {"env": env, "dev_ms": {}}
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
        t = {}
        tag = f"{rows}x{W}"
        # byte accounting: idx int64 + src + inp read, out written
        t["bytes_MB"] = round((n * 8 + n * 2 * 3) / 1e6, 2)
        t["bw_cal"] = bench(lambda: k_bw[(triton.cdiv(n, 4096),)](
            idx, src, inp, out, n, BLOCK=4096))
        for blk in (1024, 2048):
            if W % blk:
                continue
            g1 = (rows, W // blk)
            t[f"atomic_nomask_b{blk}"] = bench(
                lambda: k_atomic[g1](src, idx, buf, W, BLOCK=blk, MASKED=False))
            t[f"atomic_mask_b{blk}"] = bench(
                lambda: k_atomic[g1](src, idx, buf, W, BLOCK=blk, MASKED=True))
        t["sort_f32"] = bench(lambda: k_sort_only[(rows,)](idx, keys, W, B))
        t["sort_i32"] = bench(lambda: k_sort_i32[(rows,)](idx, keys, W, B))
        t["hist"] = bench(lambda: k_hist_only[(rows,)](idx, cum, W, B))
        t["gather2"] = bench(lambda: k_gather2[(rows,)](pre, cum, pre, B))
        t["own_f32_b128c64"] = bench(lambda: k_own_f32[(rows, W // 128)](
            inp, idx, src, out, W, BLOCK=128, CHUNK=64))
        t["own_f16_b128c64"] = bench(lambda: k_own_f16[(rows, W // 128)](
            inp, idx, src, out, W, BLOCK=128, CHUNK=64))
        # correctness of the fp16-compare variant
        ref = torch.scatter_add(inp, -1, idx, src)
        k_own_f16[(rows, W // 128)](inp, idx, src, out, W, BLOCK=128, CHUNK=64)
        torch.npu.synchronize()
        d = (out.float() - ref.float()).abs()
        t["own_f16_maxabs"] = round(float(d.max().item()), 6)
        t["own_f16_ok"] = bool((d <= 0.015 + 0.015 * ref.float().abs()).all().item())
        res["dev_ms"][tag] = t
        log("dev", case=tag, **t)

    with open(os.path.join(ART, "probe_components.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("DONE", flush=True)


try:
    main()
except Exception:  # noqa: BLE001
    traceback.print_exc()
    print("FAILED", flush=True)
