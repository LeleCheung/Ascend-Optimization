"""Bounded target-side diagnostic #2 for expm1_: separate DMA floor from math cost,
test deeper pipelining, and check hardware exp2 numerics.

Kernels share one shape: unmasked main loop over full tiles + one masked tail tile,
grid = num_vectorcore (40) unless overridden.
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
OUT = os.path.join(ART_DIR, "bench2.json")
res = {"numerics": {}, "variants": [], "ir": {}}


def save():
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)


LOG2E = 1.4426950408889634


@triton.jit
def _f_exp(x):
    xf = x.to(tl.float32)
    y = tl.exp(xf) - 1.0
    return tl.where(xf == 0.0, xf, y)


@triton.jit
def _f_exp2(x):
    xf = x.to(tl.float32)
    y = tl.exp2(xf * 1.4426950408889634) - 1.0
    return tl.where(xf == 0.0, xf, y)


@triton.jit
def _f_mul(x):
    # DMA floor probe: same load/store traffic, negligible math
    xf = x.to(tl.float32)
    return xf * 1.5


@triton.jit
def k_exp(a_ptr, n, BLOCK: tl.constexpr, NS: tl.constexpr):
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    full = n // BLOCK
    for t in tl.range(pid, full, G, num_stages=NS):
        off = t * BLOCK + lane
        x = tl.load(a_ptr + off)
        y = _f_exp(x)
        tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty))
    rem = n - full * BLOCK
    if rem > 0:
        if full % G == pid:
            off = full * BLOCK + lane
            m = off < n
            x = tl.load(a_ptr + off, mask=m, other=0.0)
            y = _f_exp(x)
            tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty), mask=m)


@triton.jit
def k_exp2(a_ptr, n, BLOCK: tl.constexpr, NS: tl.constexpr):
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    full = n // BLOCK
    for t in tl.range(pid, full, G, num_stages=NS):
        off = t * BLOCK + lane
        x = tl.load(a_ptr + off)
        y = _f_exp2(x)
        tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty))
    rem = n - full * BLOCK
    if rem > 0:
        if full % G == pid:
            off = full * BLOCK + lane
            m = off < n
            x = tl.load(a_ptr + off, mask=m, other=0.0)
            y = _f_exp2(x)
            tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty), mask=m)


@triton.jit
def k_mul(a_ptr, n, BLOCK: tl.constexpr, NS: tl.constexpr):
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    full = n // BLOCK
    for t in tl.range(pid, full, G, num_stages=NS):
        off = t * BLOCK + lane
        x = tl.load(a_ptr + off)
        y = _f_mul(x)
        tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty))
    rem = n - full * BLOCK
    if rem > 0:
        if full % G == pid:
            off = full * BLOCK + lane
            m = off < n
            x = tl.load(a_ptr + off, mask=m, other=0.0)
            y = _f_mul(x)
            tl.store(a_ptr + off, y.to(a_ptr.dtype.element_ty), mask=m)


def bench(fn, iters=100, warmup=15):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def add(name, fn):
    try:
        res["variants"].append({"name": name, "wall_ms_async": bench(fn)})
    except Exception as e:  # noqa: BLE001
        res["variants"].append({"name": name, "error": repr(e)[:300]})
    save()


G = 40


def check(tag, kernel, dtype, block, tols):
    torch.manual_seed(0)
    n = (1 << 20) + 12345
    x0 = torch.randn(n, dtype=torch.float32)
    x0[:7] = torch.tensor([0.0, -0.0, float("nan"), float("inf"), -float("inf"), -1.0, 1.0])
    a = x0.to(dtype=dtype, device="npu")
    ref = a.clone()
    ref.expm1_()
    cand = a.clone()
    kernel[(G,)](cand, n, BLOCK=block, NS=2)
    torch.npu.synchronize()
    r = ref.float()
    c = cand.float()
    atol, rtol = tols
    close = torch.isclose(c, r, atol=atol, rtol=rtol, equal_nan=True)
    diff = (c - r).abs()
    rec = {
        "dtype": str(dtype), "bad": int((~close).sum().item()), "n": n,
        "max_abs_err": float(diff.max().item()),
        "max_rel_err": float((diff / r.abs().clamp_min(1e-30)).max().item()),
        "special_out": [float(v) for v in c[:7].tolist()],
        "neg_zero_signbit_cand": bool(torch.signbit(cand[1].float()).item()),
        "neg_zero_signbit_ref": bool(torch.signbit(ref[1].float()).item()),
    }
    band = (r.abs() > 1e-6) & (r.abs() < 1e-2)
    if bool(band.any()):
        rec["band_max_rel_err"] = float((diff[band] / r[band].abs()).max().item())
    if rec["bad"]:
        idx = (~close).nonzero()[:6].flatten().tolist()
        rec["bad_samples"] = [[float(c[i]), float(r[i])] for i in idx]
    res["numerics"][tag] = rec
    save()


# ---- numerics of both formulas (also warms their kernels)
for tag, kern in (("exp_fp32", k_exp), ("exp2_fp32", k_exp2)):
    try:
        check(tag, kern, torch.float32, 4096, (1e-4, 1e-4))
    except Exception as e:  # noqa: BLE001
        res["numerics"][tag + "_error"] = repr(e)[:300]
        save()
for tag, kern in (("exp_fp16", k_exp), ("exp2_fp16", k_exp2)):
    try:
        check(tag, kern, torch.float16, 8192, (1.5e-2, 1.5e-2))
    except Exception as e:  # noqa: BLE001
        res["numerics"][tag + "_error"] = repr(e)[:300]
        save()

# ---- IR excerpt of the fp32 exp kernel (is exp a call or a poly?)
try:
    ck = k_exp.warmup(torch.empty(1, device="npu", dtype=torch.float32), 1, BLOCK=4096, NS=2, grid=(1,))
    asm = ck.asm
    res["ir"]["asm_keys"] = list(asm.keys())
    ttir = asm.get("ttir", "")
    lines = [ln.strip() for ln in ttir.splitlines() if ("exp" in ln or "arith.select" in ln or "cmp" in ln)]
    res["ir"]["ttir_exp_lines"] = lines[:25]
    res["ir"]["ttir_len"] = len(ttir)
except Exception as e:  # noqa: BLE001
    res["ir"]["error"] = repr(e)[:300]
save()

# ---- timing: fp32 4096x4096
n32 = 4096 * 4096
A32 = torch.randn(n32, dtype=torch.float32, device="npu")
add("fp32|exp_ns2_g40", lambda: k_exp[(G,)](A32, n32, BLOCK=4096, NS=2))
add("fp32|mul_ns2_g40", lambda: k_mul[(G,)](A32, n32, BLOCK=4096, NS=2))
add("fp32|exp2_ns2_g40", lambda: k_exp2[(G,)](A32, n32, BLOCK=4096, NS=2))
add("fp32|exp_ns3_g40", lambda: k_exp[(G,)](A32, n32, BLOCK=4096, NS=3))
add("fp32|exp_ns4_g40", lambda: k_exp[(G,)](A32, n32, BLOCK=4096, NS=4))
add("fp32|exp2_ns4_g40", lambda: k_exp2[(G,)](A32, n32, BLOCK=4096, NS=4))
add("fp32|mul_ns4_g40", lambda: k_mul[(G,)](A32, n32, BLOCK=4096, NS=4))
add("fp32|exp_ns2_g80", lambda: k_exp[(G * 2,)](A32, n32, BLOCK=4096, NS=2))
add("fp32|exp_ns2_g20", lambda: k_exp[(G // 2,)](A32, n32, BLOCK=4096, NS=2))
del A32
save()

# ---- timing: fp16 4096x4096
n16 = 4096 * 4096
A16 = torch.randn(n16, dtype=torch.float32).to(torch.float16).npu()
add("fp16|exp_ns2_g40_B8192", lambda: k_exp[(G,)](A16, n16, BLOCK=8192, NS=2))
add("fp16|mul_ns2_g40_B8192", lambda: k_mul[(G,)](A16, n16, BLOCK=8192, NS=2))
add("fp16|exp2_ns2_g40_B8192", lambda: k_exp2[(G,)](A16, n16, BLOCK=8192, NS=2))
add("fp16|exp_ns4_g40_B8192", lambda: k_exp[(G,)](A16, n16, BLOCK=8192, NS=4))
add("fp16|mul_ns4_g40_B8192", lambda: k_mul[(G,)](A16, n16, BLOCK=8192, NS=4))
add("fp16|exp_ns2_g80_B8192", lambda: k_exp[(G * 2,)](A16, n16, BLOCK=8192, NS=2))
del A16
save()

# ---- 1M-element case (medium) with the best structure
n1 = 1 << 20
A1 = torch.randn(n1, dtype=torch.float32, device="npu")
add("fp32_1M|exp_ns2_g40", lambda: k_exp[(G,)](A1, n1, BLOCK=4096, NS=2))
add("fp32_1M|exp_ns2_g16", lambda: k_exp[(16,)](A1, n1, BLOCK=4096, NS=2))
del A1
save()

print(json.dumps(res, indent=1)[:8000])
