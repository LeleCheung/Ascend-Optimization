"""Bounded target-side geometry sweep for moe_sum_reduce (diagnostic only)."""
import json
import os
import time

import torch
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
RESULT = {"probe": {}, "seed": {}, "sweep": {}, "persist": {}, "correct": {}}
T0 = time.time()


def elapsed():
    return time.time() - T0


def rec(bucket, key, val):
    RESULT[bucket][key] = val
    print(bucket, key, val, flush=True)


# ---------------------------------------------------------------- kernels
@triton.jit
def _k_noop(x_ptr):
    offs = tl.arange(0, 128)
    tl.store(x_ptr + offs, tl.zeros((128,), dtype=tl.float32))


@triton.jit
def _k_static(in_ptr, out_ptr, scale,
              K: tl.constexpr, H: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr,
              EVEN_M: tl.constexpr, EVEN_N: tl.constexpr,
              T: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    in_off = offs_m[:, None] * (K * H) + offs_n[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if EVEN_M and EVEN_N:
        for k in range(K):
            acc += tl.load(in_ptr + in_off + k * H).to(tl.float32)
        tl.store(out_ptr + offs_m[:, None] * H + offs_n[None, :], acc * scale)
    else:
        mask = (offs_m[:, None] < T) & (offs_n[None, :] < H)
        for k in range(K):
            acc += tl.load(in_ptr + in_off + k * H, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs_m[:, None] * H + offs_n[None, :], acc * scale, mask=mask)


@triton.jit
def _k_persist(in_ptr, out_ptr, scale,
               K: tl.constexpr, H: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr,
               NBH: tl.constexpr, TOTAL: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offs_m = tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    for i in range(pid, TOTAL, nprog):
        blk = i % NBH
        row = (i // NBH) * BM
        toks = row + offs_m
        col = blk * BN + offs_n
        in_off = toks[:, None] * (K * H) + col[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(K):
            acc += tl.load(in_ptr + in_off + k * H).to(tl.float32)
        tl.store(out_ptr + toks[:, None] * H + col[None, :], acc * scale)


@triton.jit
def _k_seed(input_ptr, output_ptr, num_tokens: tl.constexpr, top_k: tl.constexpr,
            hidden_dim: tl.constexpr, s0, s1, s2, o0, o1, scale,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    om = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    on = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (om[:, None] < num_tokens) & (on[None, :] < hidden_dim)
    io = om[:, None] * s0 + on[None, :] * s2
    red = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, top_k):
        red += tl.load(input_ptr + io + k * s1, mask=mask, other=0.0).to(tl.float32)
    red *= scale
    oo = om[:, None] * o0 + on[None, :] * o1
    tl.store(output_ptr + oo, red, mask=mask)


# ---------------------------------------------------------------- bench
def bench(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def ref_fn(x, s):
    return x.float().sum(dim=1).mul(s).to(x.dtype)


def make(T, K=8, H=7168):
    x = torch.randn(T, K, H, device="npu", dtype=torch.float32).to(torch.bfloat16)
    out = torch.empty(T, H, device="npu", dtype=torch.bfloat16)
    return x, out


def run_static(x, out, T, K, H, BM, BN, warps, stages):
    nm = -(-T // BM)
    nn = -(-H // BN)
    _k_static[(nm, nn)](x, out, 2.5, K=K, H=H, BM=BM, BN=BN,
                        EVEN_M=(T % BM == 0), EVEN_N=(H % BN == 0), T=T,
                        num_warps=warps, num_stages=stages)


def run_persist(x, out, T, K, H, BM, BN, nprog, warps, stages):
    nbh = H // BN
    total = (T // BM) * nbh
    nprog = min(nprog, total)
    _k_persist[(nprog,)](x, out, 2.5, K=K, H=H, BM=BM, BN=BN, NBH=nbh,
                         TOTAL=total, num_warps=warps, num_stages=stages)


# ---------------------------------------------------------------- probes
try:
    from triton.backends import backends
    from triton.backends.compiler import GPUTarget  # noqa
except Exception:
    pass
try:
    from triton.runtime import driver
    props = driver.active.utils.get_device_properties(torch.npu.current_device())
    rec("probe", "device_props", {k: v for k, v in dict(props).items()})
except Exception as exc:  # noqa
    rec("probe", "device_props_error", repr(exc))

rec("probe", "torch", torch.__version__)
rec("probe", "triton", triton.__version__)
try:
    rec("probe", "torch_npu", __import__("torch_npu").__version__)
except Exception as exc:  # noqa
    rec("probe", "torch_npu", repr(exc))

# fixed launch overhead of a trivial kernel
x0 = torch.empty(128, device="npu", dtype=torch.float32)
_k_noop[(1,)](x0)
rec("probe", "noop_launch_ms", bench(lambda: _k_noop[(1,)](x0), 200))
ck = None
try:
    caches = getattr(_k_noop, "device_caches", None)
    if caches:
        for v in caches.values():
            d = v[0] if isinstance(v, tuple) else v
            if isinstance(d, dict) and d:
                ck = list(d.values())[0]
    rec("probe", "compiled_kernel_type", type(ck).__name__)
    rec("probe", "compiled_has_getitem", hasattr(ck, "__getitem__"))
    if hasattr(ck, "__getitem__"):
        runner = ck[(1, 1, 1)]
        runner(x0)
        rec("probe", "direct_launch_ms", bench(lambda: runner(x0), 200))
except Exception as exc:  # noqa
    rec("probe", "direct_launch_error", repr(exc))

# ---------------------------------------------------------------- seed validation
shape_list = [1, 8, 64, 512, 4096]
for T in shape_list:
    x, out = make(T)
    if T <= 64:
        bm, bn, w = 2, 512, 8
    else:
        bm, bn, w = 1, 2048, 2
    bn = min(bn, 8192)
    f = lambda: _k_seed[(-(-T // bm), -(-7168 // bn))](
        x, out, T, 8, 7168, *x.stride(), *out.stride(), 2.5,
        BLOCK_M=bm, BLOCK_N=bn, num_warps=w, num_stages=1)
    rec("seed", f"T{T}", bench(f, 30))
    rec("seed", f"T{T}_ref", bench(lambda: ref_fn(x, 2.5), 30))
    del x, out

# ---------------------------------------------------------------- static sweep
SWEEP = [(1, 512), (1, 1024), (1, 2048), (2, 1024), (2, 2048), (4, 1024),
         (4, 2048), (8, 512), (8, 1024), (16, 512), (16, 1024), (32, 512)]
BIG = 4096
x, out = make(BIG)
best = []
for bm, bn in SWEEP:
    if elapsed() > 150:
        rec("sweep", "aborted_early", True)
        break
    ref = ref_fn(x, 2.5)
    out.zero_()
    try:
        run_static(x, out, BIG, 8, 7168, bm, bn, 4, 1)
        torch.npu.synchronize()
        err = (out.float() - ref.float()).abs().max().item()
        t = bench(lambda: run_static(x, out, BIG, 8, 7168, bm, bn, 4, 1), 20)
        gb = 528.5 / t
        rec("sweep", f"bm{bm}_bn{bn}_w4_s1", {"ms": t, "GBps": gb, "err": err})
        best.append((t, bm, bn, 4, 1))
    except Exception as exc:  # noqa
        rec("sweep", f"bm{bm}_bn{bn}_w4_s1_ERROR", repr(exc)[:300])
best.sort()
top = best[:2]

# warps / stages refinement on the best geometry
for _, bm, bn, _, _ in top:
    for w in (1, 2, 8):
        try:
            t = bench(lambda: run_static(x, out, BIG, 8, 7168, bm, bn, w, 1), 20)
            rec("sweep", f"bm{bm}_bn{bn}_w{w}_s1", {"ms": t, "GBps": 528.5 / t})
            best.append((t, bm, bn, w, 1))
        except Exception as exc:  # noqa
            rec("sweep", f"bm{bm}_bn{bn}_w{w}_s1_ERROR", repr(exc)[:300])
    for s in (2, 3):
        try:
            t = bench(lambda: run_static(x, out, BIG, 8, 7168, bm, bn, 4, s), 20)
            rec("sweep", f"bm{bm}_bn{bn}_w4_s{s}", {"ms": t, "GBps": 528.5 / t})
            best.append((t, bm, bn, 4, s))
        except Exception as exc:  # noqa
            rec("sweep", f"bm{bm}_bn{bn}_w4_s{s}_ERROR", repr(exc)[:300])
best.sort()

# ---------------------------------------------------------------- persistent
if elapsed() < 190:
    for bm, bn in [(1, 1024), (4, 1024), (8, 1024), (4, 2048)]:
        for nprog in (20, 40, 80, 160, 320):
            if elapsed() > 220:
                break
            try:
                t = bench(lambda: run_persist(x, out, BIG, 8, 7168, bm, bn, nprog, 4, 1), 20)
                rec("persist", f"bm{bm}_bn{bn}_np{nprog}", {"ms": t, "GBps": 528.5 / t})
            except Exception as exc:  # noqa
                rec("persist", f"bm{bm}_bn{bn}_np{nprog}_ERROR", repr(exc)[:300])
del x, out

# ---------------------------------------------------------------- small shapes
cands = []
for _, bm, bn, w, s in best[:3]:
    cands.append(("static", bm, bn, w, s))
cands.append(("static", 2, 512, 8, 1))
cands.append(("static", 1, 1024, 4, 1))
for T in (1, 8, 64, 512):
    if elapsed() > 240:
        break
    x, out = make(T)
    rec("sweep", f"T{T}_ref", bench(lambda: ref_fn(x, 2.5), 30))
    for kind, bm, bn, w, s in cands:
        if T % bm != 0:
            continue
        f = lambda: run_static(x, out, T, 8, 7168, bm, bn, w, s)
        try:
            rec("sweep", f"T{T}_bm{bm}_bn{bn}_w{w}_s{s}", bench(f, 30))
        except Exception as exc:  # noqa
            rec("sweep", f"T{T}_bm{bm}_bn{bn}_w{w}_s{s}_ERROR", repr(exc)[:200])
    del x, out

# ---------------------------------------------------------------- correctness corner shapes
for (T, K, H) in [(7, 2, 128), (83, 4, 512), (3, 8, 2048)]:
    x, out = make(T, K, H)
    ref = ref_fn(x, 2.5)
    for bm, bn, w in [(2, 512, 8), (1, 128, 4)]:
        bn2 = min(bn, triton.next_power_of_2(H))
        try:
            run_static(x, out, T, K, H, bm, bn2, w, 1)
            torch.npu.synchronize()
            err = (out.float() - ref.float()).abs().max().item()
            rec("correct", f"T{T}_K{K}_H{H}_bm{bm}_bn{bn2}", err)
        except Exception as exc:  # noqa
            rec("correct", f"T{T}_K{K}_H{H}_bm{bm}_bn{bn2}_ERROR", repr(exc)[:200])
    del x, out

RESULT["elapsed_s"] = elapsed()
with open(os.path.join(ART, "sweep.json"), "w") as fh:
    json.dump(RESULT, fh, indent=1)
print("done", RESULT["elapsed_s"], flush=True)
