"""Bounded target-side diagnostic for expm1_ (debug job, not a formal round).

1. Prints device properties (AIV/vector core count) used for grid sizing.
2. Numerics of the light formula (exp(x)-1 + signed-zero select) vs torch.expm1_
   under the frozen tolerances, including the special-value pattern and a tail tile.
3. Times candidate loop structures for the two 4096x4096 cases.
"""
import json
import os
import time

import torch
import torch_npu
import triton
import triton.language as tl

ART_DIR = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
os.makedirs(ART_DIR, exist_ok=True)
OUT = os.path.join(ART_DIR, "bench_expm1.json")
res = {"facts": {}, "numerics": {}, "variants": []}


def save():
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)


# ---------------------------------------------------------------- device facts
try:
    from triton.runtime import driver as _driver
    dev = torch.npu.current_device()
    dp = _driver.active.utils.get_device_properties(dev)
    try:
        d = {k: v for k, v in dict(dp).items() if isinstance(v, (int, float, str))}
    except Exception:
        d = {k: getattr(dp, k) for k in dir(dp)
             if not k.startswith("_") and isinstance(getattr(dp, k), (int, float, str))}
    res["facts"]["device_properties"] = d
except Exception as e:  # noqa: BLE001
    res["facts"]["device_properties_error"] = repr(e)
try:
    res["facts"]["npu_device_name"] = torch.npu.get_device_name(0)
except Exception as e:  # noqa: BLE001
    res["facts"]["npu_device_name_error"] = repr(e)
save()


# ---------------------------------------------------------------- value formulas
@triton.jit
def _f_light(x):
    xf = x.to(tl.float32)
    y = tl.exp(xf) - 1.0
    return tl.where(xf == 0.0, xf, y)


@triton.jit
def _f_heavy(x):
    xf = x.to(tl.float32)
    series = xf * (1.0 + 0.5 * xf)
    value = tl.where(tl.abs(xf) < 0.01, series, tl.exp(xf) - 1.0)
    bits = tl.where(xf == 0, xf.to(tl.uint32, bitcast=True), value.to(tl.uint32, bitcast=True))
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def k_stage(a_ptr, n, BLOCK: tl.constexpr, NS: tl.constexpr):
    """Unmasked main loop over full tiles + one masked tail tile."""
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    full = n // BLOCK
    for t in tl.range(pid, full, G, num_stages=NS):
        off = t * BLOCK + lane
        x = tl.load(a_ptr + off)
        y = _f_light(x)
        tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty))
    rem = n - full * BLOCK
    if rem > 0:
        if full % G == pid:
            off = full * BLOCK + lane
            m = off < n
            x = tl.load(a_ptr + off, mask=m, other=0.0)
            y = _f_light(x)
            tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty), mask=m)


@triton.jit
def k_stage_heavy(a_ptr, n, BLOCK: tl.constexpr, NS: tl.constexpr):
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    full = n // BLOCK
    for t in tl.range(pid, full, G, num_stages=NS):
        off = t * BLOCK + lane
        x = tl.load(a_ptr + off)
        y = _f_heavy(x)
        tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty))
    rem = n - full * BLOCK
    if rem > 0:
        if full % G == pid:
            off = full * BLOCK + lane
            m = off < n
            x = tl.load(a_ptr + off, mask=m, other=0.0)
            y = _f_heavy(x)
            tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty), mask=m)


@triton.jit
def k_flat(a_ptr, n, BLOCK: tl.constexpr):
    """One tile per program, masked (seed-style flat kernel)."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    x = tl.load(a_ptr + off, mask=m, other=0.0)
    y = _f_light(x)
    tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty), mask=m)


# ---------------------------------------------------------------- timing helpers
def bench(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def bench_sync(fn, iters=60, warmup=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
        torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def add_variant(name, fn):
    try:
        lat = bench(fn)
        lat_s = bench_sync(fn)
        res["variants"].append({"name": name, "wall_ms_async": lat, "wall_ms_sync": lat_s})
    except Exception as e:  # noqa: BLE001
        res["variants"].append({"name": name, "error": repr(e)[:400]})
    save()


G = int(res["facts"].get("device_properties", {}).get("num_vectorcore", 40) or 40)


# ---------------------------------------------------------------- numerics check
def check(tag, dtype, block, tols):
    torch.manual_seed(0)
    n = (1 << 20) + 12345
    x0 = torch.randn(n, dtype=torch.float32)
    edge = torch.tensor([0.0, -0.0, float("nan"), float("inf"), -float("inf"), -1.0, 1.0])
    x0[:7] = edge
    a = x0.to(dtype=dtype, device="npu")
    ref = a.clone()
    ref.expm1_()
    cand = a.clone()
    k_stage[(G,)](cand, n, BLOCK=block, NS=1)
    torch.npu.synchronize()
    r = ref.float()
    c = cand.float()
    atol, rtol = tols
    close = torch.isclose(c, r, atol=atol, rtol=rtol, equal_nan=True)
    bad = int((~close).sum().item())
    diff = (c - r).abs()
    rec = {
        "dtype": str(dtype), "n": n, "block": block, "bad": bad,
        "matched": int(close.sum().item()),
        "max_abs_err": float(diff.max().item()),
        "max_rel_err": float((diff / r.abs().clamp_min(1e-30)).max().item()),
        "special_out": [float(v) for v in c[:7].tolist()],
        "special_ref": [float(v) for v in r[:7].tolist()],
        "neg_zero_signbit_cand": bool(torch.signbit(cand[1].float()).item()),
        "neg_zero_signbit_ref": bool(torch.signbit(ref[1].float()).item()),
    }
    band = (r.abs() > 1e-6) & (r.abs() < 1e-2)
    if bool(band.any()):
        rec["band_max_rel_err"] = float((diff[band] / r[band].abs()).max().item())
        rec["band_max_abs_err"] = float(diff[band].max().item())
    if bad:
        idx = (~close).nonzero()[:8].flatten().tolist()
        rec["bad_samples"] = [[float(c[i]), float(r[i])] for i in idx]
    return rec


try:
    res["numerics"]["float32"] = check("light", torch.float32, 8192, (1e-4, 1e-4))
except Exception as e:  # noqa: BLE001
    res["numerics"]["float32_error"] = repr(e)[:400]
save()
try:
    res["numerics"]["float16"] = check("light", torch.float16, 16384, (1.5e-2, 1.5e-2))
except Exception as e:  # noqa: BLE001
    res["numerics"]["float16_error"] = repr(e)[:400]
save()

# ---------------------------------------------------------------- timing variants
cases = [
    ("fp32_4096x4096", torch.float32, 4096 * 4096, 8192),
    ("fp16_4096x4096", torch.float16, 4096 * 4096, 16384),
]


def mk(kernel, A, n, block, grid, ns=None):
    if ns is None:
        return lambda: kernel[(grid,)](A, n, BLOCK=block)
    return lambda: kernel[(grid,)](A, n, BLOCK=block, NS=ns)


for label, dtype, n, big in cases:
    A = torch.randn(n, dtype=torch.float32).to(dtype=dtype, device="npu")
    add_variant(f"{label}|flat_light_B4096", mk(k_flat, A, n, 4096, (n + 4095) // 4096))
    add_variant(f"{label}|stage_B{big}_ns1_g{G}", mk(k_stage, A, n, big, G, 1))
    add_variant(f"{label}|stage_B{big}_ns2_g{G}", mk(k_stage, A, n, big, G, 2))
    if dtype is torch.float32:
        add_variant(f"{label}|stage_B{big}_ns3_g{G}", mk(k_stage, A, n, big, G, 3))
    add_variant(f"{label}|stage_B{big // 2}_ns2_g{G}", mk(k_stage, A, n, big // 2, G, 2))
    if dtype is torch.float32:
        add_variant(f"{label}|stage_heavy_B{big}_ns2_g{G}",
                    mk(k_stage_heavy, A, n, big, G, 2))
    add_variant(f"{label}|stage_B{big}_ns2_g{2 * G}", mk(k_stage, A, n, big, 2 * G, 2))
    add_variant(f"{label}|stage_B{big}_ns2_g{4 * G}", mk(k_stage, A, n, big, 4 * G, 2))
    del A
    save()

save()
print(json.dumps(res, indent=1)[:6000])
