"""Measure host-side launch overhead sensitivity: arg count, grid, wrapper cost."""
import json
import os
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEV = "npu:0"
ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")


def bench(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


@triton.jit
def s1(p):
    tl.store(p + tl.arange(0, 1), 0.0)


@triton.jit
def s2(p, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    tl.store(p + pid * BLOCK + tl.arange(0, BLOCK), tl.zeros([BLOCK], dtype=p.dtype.element_ty))


@triton.jit
def s3(p, n, nt):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    a = tl.arange(0, 8192)
    z = tl.zeros([8192], dtype=p.dtype.element_ty)
    for t in range(pid, nt, nprog):
        tl.store(p + t * 8192 + a, z)


@triton.jit
def s4(out_ptr, g_ptr, i_ptr, outer, red, gs_o, is_o, n_tiles, nb,
       BLOCK_O: tl.constexpr, BLOCK_R: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    ar = tl.arange(0, BLOCK_R)
    ao = tl.arange(0, BLOCK_O)
    for t in range(pid, n_tiles, nprog):
        nb_ = nb
        o = (t // nb_) * BLOCK_O + ao
        rb = t % nb_
        r = rb * BLOCK_R + ar
        v = tl.load(i_ptr + o * is_o, mask=o < outer, other=-1).to(tl.int32)
        g = tl.load(g_ptr + o * gs_o, mask=o < outer, other=0.0)
        cond = r[None, :] == v[:, None]
        g2 = tl.broadcast_to(g[:, None], (BLOCK_O, BLOCK_R))
        val = tl.where(cond, g2, tl.zeros_like(g2))
        tl.store(out_ptr + o[:, None] * red + r[None, :], val, mask=(o < outer)[:, None])


@triton.jit
def s5(out_ptr, g_ptr, i_ptr, OUTER: tl.constexpr, RED: tl.constexpr,
       GSO: tl.constexpr, ISO: tl.constexpr, NB: tl.constexpr,
       NT: tl.constexpr, BLOCK_O: tl.constexpr, BLOCK_R: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    ar = tl.arange(0, BLOCK_R)
    ao = tl.arange(0, BLOCK_O)
    for t in range(pid, NT, nprog):
        o = (t // NB) * BLOCK_O + ao
        rb = t % NB
        r = rb * BLOCK_R + ar
        v = tl.load(i_ptr + o * ISO, mask=o < OUTER, other=-1).to(tl.int32)
        g = tl.load(g_ptr + o * GSO, mask=o < OUTER, other=0.0)
        cond = r[None, :] == v[:, None]
        g2 = tl.broadcast_to(g[:, None], (BLOCK_O, BLOCK_R))
        val = tl.where(cond, g2, tl.zeros_like(g2))
        tl.store(out_ptr + o[:, None] * RED + r[None, :], val, mask=(o < OUTER)[:, None])


res = {}
N = 1024 * 1024
buf = torch.empty(N, dtype=torch.float16, device=DEV)
g = torch.randn((1024,), dtype=torch.float16, device=DEV)
idx = torch.randint(1024, (1024,), device=DEV)
sizes = [1024, 1024]
out = torch.empty(sizes, dtype=torch.float16, device=DEV)

res["s1_g1_1arg"] = round(bench(lambda: s1[(1,)](buf)), 4)
res["s2_g128_2arg"] = round(bench(lambda: s2[(128,)](buf, BLOCK=8192)), 4)
res["s2_g512_2arg"] = round(bench(lambda: s2[(512,)](buf, BLOCK=8192)), 4)
res["s3_g128_3arg"] = round(bench(lambda: s3[(128,)](buf, N, N // 8192)), 4)
res["s3_g512_3arg"] = round(bench(lambda: s3[(512,)](buf, N, N // 8192)), 4)

# realistic inner==1 case: outer=1024, red=1024
res["s4_g128_9arg"] = round(bench(lambda: s4[(128,)](out, g, idx, 1024, 1024, 1, 1, 1024, 1,
                                                       BLOCK_O=1, BLOCK_R=1024)), 4)
res["s4_g128_9arg_pos"] = round(bench(lambda: s4[(128,)](out, g, idx, 1024, 1024, 1, 1, 1024, 1,
                                                           1, 1024)), 4)
res["s5_g128_3arg"] = round(bench(lambda: s5[(128,)](out, g, idx, 1024, 1024, 1, 1, 1, 1024,
                                                     1, 1024)), 4)
res["s4_g128_nw1"] = round(bench(lambda: s4[(128,)](out, g, idx, 1024, 1024, 1, 1, 1024, 1,
                                                    BLOCK_O=1, BLOCK_R=1024, num_warps=1)), 4)

ref = torch.ops.aten.value_selecting_reduction_backward(g, 1, idx, sizes, False)
s5[(128,)](out, g, idx, 1024, 1024, 1, 1, 1, 1024, 1, 1024)
torch.npu.synchronize()
res["s5_match"] = bool(torch.equal(out, ref))
s4[(128,)](out, g, idx, 1024, 1024, 1, 1, 1024, 1, BLOCK_O=1, BLOCK_R=1024)
torch.npu.synchronize()
res["s4_match"] = bool(torch.equal(out, ref))


# ---- wrapper cost breakdown ----
def w_alloc_only():
    return torch.empty(sizes, dtype=torch.float16, device=DEV)


def w_alloc_launch():
    o = torch.empty(sizes, dtype=torch.float16, device=DEV)
    s4[(128,)](o, g, idx, 1024, 1024, 1, 1, 1024, 1, BLOCK_O=1, BLOCK_R=1024)
    return o


def w_alloc_launch_const():
    o = torch.empty(sizes, dtype=torch.float16, device=DEV)
    s5[(128,)](o, g, idx, 1024, 1024, 1, 1, 1, 1024, 1, 1024)
    return o


def w_full():
    ndim = 2
    outer = 1024
    inner = 1
    o = torch.empty([1024, 1024], dtype=g.dtype, device=g.device)
    gs_o = g.stride(0)
    is_o = idx.stride(0)
    s4[(128,)](o, g, idx, outer, 1024, gs_o, is_o, 1024, 1, BLOCK_O=1, BLOCK_R=1024)
    return o


res["w_alloc_only"] = round(bench(w_alloc_only), 4)
res["w_alloc_launch"] = round(bench(w_alloc_launch), 4)
res["w_alloc_launch_const"] = round(bench(w_alloc_launch_const), 4)
res["w_full"] = round(bench(w_full), 4)
res["aten"] = round(bench(
    lambda: torch.ops.aten.value_selecting_reduction_backward(g, 1, idx, sizes, False)), 4)

with open(os.path.join(ART, "host.json"), "w") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res, indent=1))
