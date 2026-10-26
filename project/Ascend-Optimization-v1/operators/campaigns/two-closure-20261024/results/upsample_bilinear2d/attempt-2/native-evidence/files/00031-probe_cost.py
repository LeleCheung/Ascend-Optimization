"""Cost decomposition probe for the upsample_bilinear2d fast path.

Independent variants are timed back-to-back on the same geometries so that the
dispatch/gather/store cost of the current kernel can be separated.  Every
variant except the two "only" ablations is numerically checked against the
aten reference.  Artifacts are written incrementally, so a later trap still
leaves the earlier rows on disk.
"""

import json
import os
import statistics
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = os.path.join(ART, "probe_cost.json")
res = {"env": {}, "rows": [], "errors": []}


def flush():
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)


def note(msg):
    res["errors"].append(msg)
    flush()


try:
    props = torch.npu.get_device_properties(torch.npu.current_device())
    res["env"]["props"] = {k: str(getattr(props, k)) for k in dir(props) if not k.startswith("_")}
except Exception as exc:  # pragma: no cover
    res["env"]["props"] = repr(exc)
try:
    from triton.runtime import driver
    res["env"]["driver_props"] = {
        k: str(v)
        for k, v in driver.active.utils.get_device_properties(torch.npu.current_device()).items()
    }
except Exception as exc:
    res["env"]["driver_props"] = repr(exc)
res["env"]["torch"] = torch.__version__
res["env"]["triton"] = triton.__version__
res["env"]["max_threads"] = str(triton.runtime.driver.active.get_current_target())
flush()


# ---------------------------------------------------------------- variants
@triton.jit
def k_parent(ptr_o, ptr_i, n_rows, s_n, s_c, s_h, rh,
             C: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
             OH: tl.constexpr, OW: tl.constexpr, BR: tl.constexpr,
             IWP: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    oh = rows % OH
    nc = rows // OH
    n = nc // C
    c = nc - n * C
    real_h = tl.maximum((oh.to(tl.float32) + 0.5) * rh - 0.5, 0.0)
    h0f = real_h.to(tl.int32)
    hl = real_h - h0f.to(tl.float32)
    h0 = tl.minimum(h0f, IH - 1)
    h1 = tl.minimum(h0 + 1, IH - 1)
    base = n * s_n + c * s_c
    rr0 = (base + h0 * s_h)[:, None]
    rr1 = (base + h1 * s_h)[:, None]
    k = tl.arange(0, IWP)
    ka = k[None, :]
    lm = tl.broadcast_to((ka < IW) & (rows[:, None] < n_rows), (BR, IWP))
    il = tl.broadcast_to(tl.maximum(k - 1, 0)[None, :], (BR, IWP))
    ir = tl.broadcast_to(tl.minimum(k + 1, IW - 1)[None, :], (BR, IWP))
    hl2 = hl[:, None]
    h0l2 = 1.0 - hl2
    v = h0l2 * tl.load(ptr_i + (rr0 + ka), mask=lm, other=0.0).to(tl.float32) \
        + hl2 * tl.load(ptr_i + (rr1 + ka), mask=lm, other=0.0).to(tl.float32)
    vm = tl.gather(v, il, 1)
    vp = tl.gather(v, ir, 1)
    even = 0.25 * vm + 0.75 * v
    odd = 0.75 * v + 0.25 * vp
    val = tl.join(even, odd)
    j = tl.arange(0, 2)
    ooff = (rows * OW)[:, None, None] + (ka * 2)[:, :, None] + j[None, None, :]
    om = tl.broadcast_to((ka < IW)[:, :, None] & (rows[:, None, None] < n_rows), (BR, IWP, 2))
    tl.store(ptr_o + ooff, val.to(ptr_o.dtype.element_ty), mask=om)


@triton.jit
def k_flat(ptr_o, ptr_i, n_rows, s_n, s_c, s_h, rh,
           C: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
           OH: tl.constexpr, OW: tl.constexpr, BR: tl.constexpr,
           NEED_MASK: tl.constexpr):
    """IW must be a power of two: flat contiguous store of BR whole rows."""
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    oh = rows % OH
    nc = rows // OH
    n = nc // C
    c = nc - n * C
    real_h = tl.maximum((oh.to(tl.float32) + 0.5) * rh - 0.5, 0.0)
    h0f = real_h.to(tl.int32)
    hl = real_h - h0f.to(tl.float32)
    h0 = tl.minimum(h0f, IH - 1)
    h1 = tl.minimum(h0 + 1, IH - 1)
    base = n * s_n + c * s_c
    rr0 = (base + h0 * s_h)[:, None]
    rr1 = (base + h1 * s_h)[:, None]
    k = tl.arange(0, IW)
    ka = k[None, :]
    hl2 = hl[:, None]
    if NEED_MASK:
        lm = tl.broadcast_to(rows[:, None] < n_rows, (BR, IW))
        v0 = tl.load(ptr_i + (rr0 + ka), mask=lm, other=0.0).to(tl.float32)
        v1 = tl.load(ptr_i + (rr1 + ka), mask=lm, other=0.0).to(tl.float32)
    else:
        v0 = tl.load(ptr_i + (rr0 + ka)).to(tl.float32)
        v1 = tl.load(ptr_i + (rr1 + ka)).to(tl.float32)
    v = (1.0 - hl2) * v0 + hl2 * v1
    il = tl.broadcast_to(tl.maximum(k - 1, 0)[None, :], (BR, IW))
    ir = tl.broadcast_to(tl.minimum(k + 1, IW - 1)[None, :], (BR, IW))
    vm = tl.gather(v, il, 1)
    vp = tl.gather(v, ir, 1)
    even = 0.25 * vm + 0.75 * v
    odd = 0.75 * v + 0.25 * vp
    val = tl.reshape(tl.join(even, odd), (BR * OW,))
    offs = pid * (BR * OW) + tl.arange(0, BR * OW)
    if NEED_MASK:
        tl.store(ptr_o + offs, val.to(ptr_o.dtype.element_ty), mask=offs < n_rows * OW)
    else:
        tl.store(ptr_o + offs, val.to(ptr_o.dtype.element_ty))


@triton.jit
def k_flat_nogather(ptr_o, ptr_i, n_rows, s_n, s_c, s_h, rh,
                    C: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
                    OH: tl.constexpr, OW: tl.constexpr, BR: tl.constexpr,
                    NEED_MASK: tl.constexpr):
    """Same as k_flat but without the two tl.gather shuffles (wrong values)."""
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    oh = rows % OH
    nc = rows // OH
    n = nc // C
    c = nc - n * C
    real_h = tl.maximum((oh.to(tl.float32) + 0.5) * rh - 0.5, 0.0)
    h0f = real_h.to(tl.int32)
    hl = real_h - h0f.to(tl.float32)
    h0 = tl.minimum(h0f, IH - 1)
    h1 = tl.minimum(h0 + 1, IH - 1)
    base = n * s_n + c * s_c
    rr0 = (base + h0 * s_h)[:, None]
    rr1 = (base + h1 * s_h)[:, None]
    k = tl.arange(0, IW)
    ka = k[None, :]
    hl2 = hl[:, None]
    if NEED_MASK:
        lm = tl.broadcast_to(rows[:, None] < n_rows, (BR, IW))
        v0 = tl.load(ptr_i + (rr0 + ka), mask=lm, other=0.0).to(tl.float32)
        v1 = tl.load(ptr_i + (rr1 + ka), mask=lm, other=0.0).to(tl.float32)
    else:
        v0 = tl.load(ptr_i + (rr0 + ka)).to(tl.float32)
        v1 = tl.load(ptr_i + (rr1 + ka)).to(tl.float32)
    v = (1.0 - hl2) * v0 + hl2 * v1
    even = 0.25 * v + 0.75 * v
    odd = 0.75 * v + 0.25 * v
    val = tl.reshape(tl.join(even, odd), (BR * OW,))
    offs = pid * (BR * OW) + tl.arange(0, BR * OW)
    if NEED_MASK:
        tl.store(ptr_o + offs, val.to(ptr_o.dtype.element_ty), mask=offs < n_rows * OW)
    else:
        tl.store(ptr_o + offs, val.to(ptr_o.dtype.element_ty))


@triton.jit
def k_store_only(ptr_o, n_rows, rh,
                 OW: tl.constexpr, BR: tl.constexpr, NEED_MASK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * (BR * OW) + tl.arange(0, BR * OW)
    val = tl.full((BR * OW,), 1.0, tl.float32) * rh
    if NEED_MASK:
        tl.store(ptr_o + offs, val.to(ptr_o.dtype.element_ty), mask=offs < n_rows * OW)
    else:
        tl.store(ptr_o + offs, val.to(ptr_o.dtype.element_ty))


@triton.jit
def k_load_only(ptr_o, ptr_i, n_rows, s_n, s_c, s_h, rh,
                C: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
                OH: tl.constexpr, BR: tl.constexpr):
    """Loads the two source rows per output row and stores one value per row."""
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    oh = rows % OH
    nc = rows // OH
    n = nc // C
    c = nc - n * C
    real_h = tl.maximum((oh.to(tl.float32) + 0.5) * rh - 0.5, 0.0)
    h0f = real_h.to(tl.int32)
    hl = real_h - h0f.to(tl.float32)
    h0 = tl.minimum(h0f, IH - 1)
    h1 = tl.minimum(h0 + 1, IH - 1)
    base = n * s_n + c * s_c
    rr0 = (base + h0 * s_h)[:, None]
    rr1 = (base + h1 * s_h)[:, None]
    k = tl.arange(0, IW)[None, :]
    hl2 = hl[:, None]
    v0 = tl.load(ptr_i + (rr0 + k)).to(tl.float32)
    v1 = tl.load(ptr_i + (rr1 + k)).to(tl.float32)
    v = (1.0 - hl2) * v0 + hl2 * v1
    tl.store(ptr_o + rows, tl.sum(v, 1).to(ptr_o.dtype.element_ty))


@triton.jit
def k_write(ptr_o, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr_o + offs, tl.full((BLOCK,), 1.0, tl.float32).to(ptr_o.dtype.element_ty),
             mask=offs < total)


# ---------------------------------------------------------------- host side
def bench(fn, reps=30, iters=5):
    fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - t0) / reps * 1e3)
    return statistics.median(samples), min(samples)


def pick_br(iw_pow2, n_rows, ub_elems=2048, target_programs=256):
    cap = max(1, ub_elems // iw_pow2)
    want = max(1, triton.cdiv(n_rows, target_programs))
    br = min(cap, want)
    br = 1 << (br.bit_length() - 1)
    while br > 1 and n_rows % br != 0:
        br >>= 1
    return max(1, br)


CASES = [
    ("t0_fp16_small", torch.float16, (1, 16, 64, 64), (128, 128)),
    ("t1_fp16_large", torch.float16, (4, 32, 128, 128), (256, 256)),
    ("t2_bf16_small", torch.bfloat16, (1, 16, 64, 64), (128, 128)),
    ("t4_fp32_small", torch.float32, (1, 16, 64, 64), (128, 128)),
    ("t5_fp32_large", torch.float32, (4, 32, 128, 128), (256, 256)),
]


def make_case(dtype, shape, out_size, seed=1730):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g).to(dtype=dtype, device="npu")
    return x


for name, dtype, shape, out_size in CASES:
    try:
        N, C, IH, IW = shape
        OH, OW = out_size
        n_rows = N * C * OH
        s_n, s_c, s_h, s_w = C * IW * IH, IW * IH, IW, 1
        rh = IH / OH
        x = make_case(dtype, shape, out_size)
        ref = torch.ops.aten.upsample_bilinear2d(x, [OH, OW], False, None, None)
        out = torch.empty((N, C, OH, OW), device="npu", dtype=dtype)
        br = pick_br(IW, n_rows)
        need_mask = (n_rows % br) != 0
        grid = (triton.cdiv(n_rows, br),)
        row = {"case": name, "dtype": str(dtype), "shape": list(shape), "out": list(out_size),
               "n_rows": n_rows, "br": br, "grid": grid[0], "need_mask": need_mask,
               "variants": []}

        def add(vname, fn, check=None):
            try:
                if check is not None:
                    check()
                med, mn = bench(fn)
                row["variants"].append({"variant": vname, "median_ms": med, "min_ms": mn})
            except Exception as exc:
                row["variants"].append({"variant": vname, "error": repr(exc)[:600]})
            flush()

        add("parent", lambda: k_parent[grid](
            out, x, n_rows, s_n, s_c, s_h, rh,
            C=C, IH=IH, IW=IW, OH=OH, OW=OW, BR=br, IWP=triton.next_power_of_2(IW)))
        row["variants"][-1]["max_abs_err"] = float((out.float() - ref.float()).abs().max())

        add("flat", lambda: k_flat[grid](
            out, x, n_rows, s_n, s_c, s_h, rh,
            C=C, IH=IH, IW=IW, OH=OH, OW=OW, BR=br, NEED_MASK=need_mask))
        row["variants"][-1]["max_abs_err"] = float((out.float() - ref.float()).abs().max())

        add("flat_nogather", lambda: k_flat_nogather[grid](
            out, x, n_rows, s_n, s_c, s_h, rh,
            C=C, IH=IH, IW=IW, OH=OH, OW=OW, BR=br, NEED_MASK=need_mask))

        add("store_only", lambda: k_store_only[grid](
            out, n_rows, rh, OW=OW, BR=br, NEED_MASK=need_mask))

        add("load_only", lambda: k_load_only[grid](
            out, x, n_rows, s_n, s_c, s_h, rh,
            C=C, IH=IH, IW=IW, OH=OH, BR=br))

        total = N * C * OH * OW
        add("write_only", lambda: k_write[(triton.cdiv(total, 4096),)](out, total, BLOCK=4096))

        # BR sweep on the flat variant
        for br2 in (4, 8, 16, 32):
            if br2 * OW > 16384:
                continue
            nm2 = (n_rows % br2) != 0
            g2 = (triton.cdiv(n_rows, br2),)
            add(f"flat_br{br2}", lambda br2=br2, nm2=nm2, g2=g2: k_flat[g2](
                out, x, n_rows, s_n, s_c, s_h, rh,
                C=C, IH=IH, IW=IW, OH=OH, OW=OW, BR=br2, NEED_MASK=nm2))

        res["rows"].append(row)
        flush()
    except Exception as exc:
        note(f"{name}: {repr(exc)[:800]}")

flush()
print(json.dumps(res, indent=1)[:4000])
