"""Bounded microbenchmark for value_selecting_reduction_backward variants."""
import json
import os
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEV = "npu:0"
ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


@triton.jit
def noop_kernel(x_ptr):
    tl.store(x_ptr + tl.arange(0, 1), 0.0)


@triton.jit
def fill_kernel(out_ptr, n, n_tiles, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    a = tl.arange(0, BLOCK)
    z = tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty)
    for t in range(pid, n_tiles, nprog):
        off = t * BLOCK + a
        if EVEN:
            tl.store(out_ptr + off, z)
        else:
            tl.store(out_ptr + off, z, mask=off < n)


@triton.jit
def fused1d_kernel(out_ptr, g_ptr, i_ptr, red, gs_o, is_o, n_tiles, nb,
                   BLOCK_R: tl.constexpr, CAST32: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    a = tl.arange(0, BLOCK_R)
    for t in range(pid, n_tiles, nprog):
        o = t // nb
        rb = t % nb
        r = rb * BLOCK_R + a
        v = tl.load(i_ptr + o * is_o)
        if CAST32:
            v = v.to(tl.int32)
        g = tl.load(g_ptr + o * gs_o)
        val = tl.where(r == v, g, tl.zeros_like(g))
        tl.store(out_ptr + o * red + r, val)


@triton.jit
def fused2d_kernel(out_ptr, g_ptr, i_ptr, outer, red, gs_o, is_o, n_tiles, nb,
                   BLOCK_O: tl.constexpr, BLOCK_R: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    ar = tl.arange(0, BLOCK_R)
    ao = tl.arange(0, BLOCK_O)
    for t in range(pid, n_tiles, nprog):
        ob = t // nb
        rb = t % nb
        o = ob * BLOCK_O + ao
        r = rb * BLOCK_R + ar
        mo = o < outer
        v = tl.load(i_ptr + o * is_o, mask=mo, other=-1).to(tl.int32)
        g = tl.load(g_ptr + o * gs_o, mask=mo, other=0.0)
        cond = r[None, :] == v[:, None]
        g2 = tl.broadcast_to(g[:, None], (BLOCK_O, BLOCK_R))
        val = tl.where(cond, g2, tl.zeros_like(g2))
        tl.store(out_ptr + o[:, None] * red + r[None, :], val, mask=mo[:, None])


@triton.jit
def scatter_kernel(out_ptr, g_ptr, i_ptr, red, gs_o, is_o, outer, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    a = pid * BLOCK + tl.arange(0, BLOCK)
    m = a < outer
    v = tl.load(i_ptr + a * is_o, mask=m, other=0).to(tl.int32)
    g = tl.load(g_ptr + a * gs_o, mask=m, other=0.0)
    tl.store(out_ptr + a * red + v, g, mask=m)


def npow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def run_case(outer, red, dtype):
    torch.manual_seed(0)
    g = torch.randn((outer,), dtype=dtype, device=DEV)
    idx = torch.randint(red, (outer,), device=DEV)
    sizes = [outer, red]
    out = torch.empty(sizes, dtype=dtype, device=DEV)
    n = outer * red
    res = {"shape": [outer, red], "dtype": str(dtype).split(".")[-1], "n": n,
           "bytes": n * out.element_size()}
    res["noop_ms"] = bench(lambda: noop_kernel[(1,)](out), iters=50)

    def ref():
        return torch.ops.aten.value_selecting_reduction_backward(g, 1, idx, sizes, False)

    res["aten_ref_ms"] = bench(ref)
    res["zero_ms"] = bench(lambda: out.zero_())
    res["zeros_alloc_ms"] = bench(lambda: torch.zeros(sizes, dtype=dtype, device=DEV))

    fill_cfgs = []
    for BLOCK in (1024, 4096, 16384):
        nt = triton.cdiv(n, BLOCK)
        for G in (32, 64, 128, 512, 4096, 65535):
            gg = min(G, nt)
            fill_cfgs.append((BLOCK, gg, nt))
    rf = {}
    for BLOCK, gg, nt in fill_cfgs:
        try:
            t = bench(lambda BLOCK=BLOCK, gg=gg, nt=nt: fill_kernel[(gg,)](
                out, n, nt, BLOCK=BLOCK, EVEN=(n % BLOCK == 0)))
            rf[f"B{BLOCK}_G{gg}"] = round(t, 4)
        except Exception as exc:  # noqa: BLE001
            rf[f"B{BLOCK}_G{gg}"] = f"ERR {type(exc).__name__}"
    res["fill"] = rf

    fz = {}
    for BLOCK in (1024, 4096):
        nb = triton.cdiv(red, BLOCK)
        nt = outer * nb
        for G in (128, 512, 4096, 65535):
            gg = min(G, nt)
            for cast in (True, False):
                key = f"B{BLOCK}_G{gg}_c{int(cast)}"
                try:
                    t = bench(lambda BLOCK=BLOCK, gg=gg, nt=nt, nb=nb, cast=cast: fused1d_kernel[(gg,)](
                        out, g, idx, red, 1, 1, nt, nb, BLOCK_R=BLOCK, CAST32=cast))
                    fz[key] = round(t, 4)
                except Exception as exc:  # noqa: BLE001
                    fz[key] = f"ERR {type(exc).__name__}"
    res["fused1d"] = fz

    f2 = {}
    for BO, BR in ((2, 512), (4, 1024), (8, 512), (8, 2048)):
        nb = triton.cdiv(red, BR)
        nt = triton.cdiv(outer, BO) * nb
        for G in (128, 512, 4096, 65535):
            gg = min(G, nt)
            key = f"O{BO}_R{BR}_G{gg}"
            try:
                t = bench(lambda BO=BO, BR=BR, gg=gg, nt=nt, nb=nb: fused2d_kernel[(gg,)](
                    out, g, idx, outer, red, 1, 1, nt, nb, BLOCK_O=BO, BLOCK_R=BR))
                f2[key] = round(t, 4)
            except Exception as exc:  # noqa: BLE001
                f2[key] = f"ERR {type(exc).__name__}"
    res["fused2d"] = f2

    # two-kernel: fill(zeros) then scatter; verify equality with the fused path
    def two_kernel(BLOCK=4096, G=512, SB=1024):
        nt = triton.cdiv(n, BLOCK)
        fill_kernel[(min(G, nt),)](out, n, nt, BLOCK=BLOCK, EVEN=(n % BLOCK == 0))
        scatter_kernel[(triton.cdiv(outer, SB),)](out, g, idx, red, 1, 1, outer, BLOCK=SB)

    res["two_kernel_ms"] = bench(two_kernel)

    # numeric check of the fastest candidates vs aten
    refv = ref()
    two_kernel()
    torch.npu.synchronize()
    res["two_kernel_match"] = bool(torch.equal(out, refv))
    BLOCK, BR = 4096, 1024
    nb = triton.cdiv(red, BR)
    nt = outer * nb
    fused1d_kernel[(min(512, nt),)](out, g, idx, red, 1, 1, nt, nb, BLOCK_R=BR, CAST32=True)
    torch.npu.synchronize()
    res["fused1d_match"] = bool(torch.equal(out, refv))
    fused2d_kernel[(min(512, triton.cdiv(outer, 4) * nb),)](out, g, idx, outer, red, 1, 1,
                                                           triton.cdiv(outer, 4) * nb, nb,
                                                           BLOCK_O=4, BLOCK_R=BR)
    torch.npu.synchronize()
    res["fused2d_match"] = bool(torch.equal(out, refv))
    return res


def main():
    out = {}
    for outer, red, dtype in (
        (8192, 8192, torch.float16),
        (8192, 8192, torch.float32),
        (1024, 1024, torch.float16),
        (32768, 512, torch.float16),
    ):
        key = f"{outer}x{red}_{str(dtype).split('.')[-1]}"
        try:
            out[key] = run_case(outer, red, dtype)
        except Exception as exc:  # noqa: BLE001
            out[key] = {"fatal": f"{type(exc).__name__}: {exc}"}
        with open(os.path.join(ART, "bench.json"), "w") as fh:
            json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


main()
