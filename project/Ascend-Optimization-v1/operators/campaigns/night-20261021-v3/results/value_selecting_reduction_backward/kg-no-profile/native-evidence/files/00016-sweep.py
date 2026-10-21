"""Sweep the contiguous row-tile kernel design over every timing shape."""
import json
import os
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEV = "npu:0"
ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
MAXP = 65535


def bench(fn, iters=15, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


@triton.jit
def row_kernel(out_ptr, g_ptr, i_ptr, outer, red, gs_o, is_o, n_tiles, nb,
               BLOCK_O: tl.constexpr, BLOCK_R: tl.constexpr,
               EVEN_O: tl.constexpr, EVEN_R: tl.constexpr):
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
        off = o[:, None] * red + r[None, :]
        if EVEN_O and EVEN_R:
            tl.store(out_ptr + off, val)
        elif EVEN_R:
            tl.store(out_ptr + off, val, mask=mo[:, None])
        else:
            tl.store(out_ptr + off, val, mask=mo[:, None] & (r[None, :] < red))


def npow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def plan(outer, red, tile):
    if red >= tile:
        BR, BO = tile, 1
    else:
        BR = npow2(red)
        BO = max(1, tile // BR)
    nb = triton.cdiv(red, BR)
    nt = triton.cdiv(outer, BO) * nb
    return BO, BR, nb, nt


def launch(out, g, idx, outer, red, tile, G):
    BO, BR, nb, nt = plan(outer, red, tile)
    grid = (min(nt, G, MAXP),)
    row_kernel[grid](out, g, idx, outer, red, 1, 1, nt, nb,
                     BLOCK_O=BO, BLOCK_R=BR,
                     EVEN_O=(outer % BO == 0), EVEN_R=(red == BR))
    return BO, BR, nb, nt


def run_new(g, idx, outer, red, tile, G, sizes, dtype):
    out = torch.empty(sizes, dtype=dtype, device=DEV)
    launch(out, g, idx, outer, red, tile, G)
    return out


CASES = []
for dt in (torch.float16, torch.bfloat16, torch.float32):
    for outer, red in ((1024, 1024), (4096, 4096), (8192, 8192),
                       (2048, 128), (8192, 256), (32768, 512)):
        CASES.append((outer, red, dt))

res = {}
for outer, red, dt in CASES:
    dn = str(dt).split(".")[-1]
    key = f"{outer}x{red}_{dn}"
    sizes = [outer, red]
    g = torch.randn((outer,), dtype=dt, device=DEV)
    idx = torch.randint(red, (outer,), device=DEV)
    out = torch.empty(sizes, dtype=dt, device=DEV)
    entry = {"n": outer * red, "bytes": outer * red * out.element_size()}
    entry["aten_ms"] = round(bench(
        lambda: torch.ops.aten.value_selecting_reduction_backward(g, 1, idx, sizes, False)), 4)
    entry["zero_ms"] = round(bench(lambda: out.zero_()), 4)
    entry["empty_ms"] = round(bench(lambda: torch.empty(sizes, dtype=dt, device=DEV)), 4)
    ref = torch.ops.aten.value_selecting_reduction_backward(g, 1, idx, sizes, False)
    sweep = {}
    for tile in (2048, 4096, 8192, 16384):
        for G in (64, 128, 256, 512, 1024):
            k = f"T{tile}_G{G}"
            try:
                launch(out, g, idx, outer, red, tile, G)
                torch.npu.synchronize()
                ok = bool(torch.equal(out, ref))
                t = bench(lambda tile=tile, G=G: launch(out, g, idx, outer, red, tile, G))
                sweep[k] = [round(t, 4), ok]
            except Exception as exc:  # noqa: BLE001
                sweep[k] = [f"ERR {type(exc).__name__}", None]
    entry["sweep"] = sweep
    # full wrapper (torch.empty + plan + launch) at the best tile/G
    best = None
    for k, v in sweep.items():
        if isinstance(v[0], float) and (best is None or v[0] < best[1]):
            best = (k, v[0])
    entry["best"] = best
    if best:
        tile = int(best[0].split("_")[0][1:])
        G = int(best[0].split("_G")[1])
        entry["wrapper_best_ms"] = round(bench(
            lambda: run_new(g, idx, outer, red, tile, G, sizes, dt)), 4)
        entry["wrapper_T8192_G512_ms"] = round(bench(
            lambda: run_new(g, idx, outer, red, 8192, 512, sizes, dt)), 4)
        entry["wrapper_T8192_G256_ms"] = round(bench(
            lambda: run_new(g, idx, outer, red, 8192, 256, sizes, dt)), 4)
    res[key] = entry
    with open(os.path.join(ART, "sweep.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    del g, idx, out, ref

print(json.dumps(res, indent=1))
