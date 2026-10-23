"""Split the residual cost into (a) the store layout and (b) per-call overhead.

t1 (8.4M outputs) is 0.629 ms of which the store path alone measured 0.293 ms;
t0 has 32x less data but is only 4.5x faster, so ~0.12 ms per call is not
device work at all.  Both are measured here.
"""

import json
import os
import sys
import traceback

import torch
import torch_npu  # noqa: F401

import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import main as cand  # noqa: E402

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
DEV = "npu:0"


@triton.jit
def k_touch(ptr_o):
    tl.store(ptr_o + tl.arange(0, 8), tl.zeros([8], dtype=tl.float32))


@triton.jit
def _rows(rows, s_n, s_c, s_h, rh, C: tl.constexpr, IH: tl.constexpr, OH: tl.constexpr):
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
    return (base + h0 * s_h)[:, None], (base + h1 * s_h)[:, None], hl[:, None]


@triton.jit
def k_flat(ptr_o, ptr_i, n_rows, s_n, s_c, s_h, rh,
           C: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
           OH: tl.constexpr, OW: tl.constexpr, BR: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rr0, rr1, hl = _rows(rows, s_n, s_c, s_h, rh, C, IH, OH)
    ka = tl.arange(0, IW)[None, :]
    h0l = 1.0 - hl
    v = h0l * tl.load(ptr_i + (rr0 + ka)).to(tl.float32) \
        + hl * tl.load(ptr_i + (rr1 + ka)).to(tl.float32)
    k = tl.arange(0, IW)
    il = tl.broadcast_to(tl.maximum(k - 1, 0)[None, :], (BR, IW))
    ir = tl.broadcast_to(tl.minimum(k + 1, IW - 1)[None, :], (BR, IW))
    vm = tl.gather(v, il, 1)
    vp = tl.gather(v, ir, 1)
    val = tl.join(0.25 * vm + 0.75 * v, 0.75 * v + 0.25 * vp)
    flat = val.reshape(BR * IW * 2)
    tl.store(ptr_o + pid * (BR * IW * 2) + tl.arange(0, BR * IW * 2),
             flat.to(ptr_o.dtype.element_ty))


@triton.jit
def k_cube(ptr_o, ptr_i, n_rows, s_n, s_c, s_h, rh,
           C: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
           OH: tl.constexpr, OW: tl.constexpr, BR: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rr0, rr1, hl = _rows(rows, s_n, s_c, s_h, rh, C, IH, OH)
    ka = tl.arange(0, IW)[None, :]
    h0l = 1.0 - hl
    v = h0l * tl.load(ptr_i + (rr0 + ka)).to(tl.float32) \
        + hl * tl.load(ptr_i + (rr1 + ka)).to(tl.float32)
    k = tl.arange(0, IW)
    il = tl.broadcast_to(tl.maximum(k - 1, 0)[None, :], (BR, IW))
    ir = tl.broadcast_to(tl.minimum(k + 1, IW - 1)[None, :], (BR, IW))
    vm = tl.gather(v, il, 1)
    vp = tl.gather(v, ir, 1)
    val = tl.join(0.25 * vm + 0.75 * v, 0.75 * v + 0.25 * vp)
    ooff = (rows * OW)[:, None, None] + (ka * 2)[:, :, None] + tl.arange(0, 2)[None, None, :]
    tl.store(ptr_o + ooff, val.to(ptr_o.dtype.element_ty))


def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    e0 = torch.npu.Event(enable_timing=True)
    e1 = torch.npu.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.npu.synchronize()
    return round(e0.elapsed_time(e1) / iters, 5)


def main():
    torch.npu.set_device(DEV)
    out = {}
    for tag, (N, C, IH, IW, OH, OW) in {
        "t0_small": (1, 16, 64, 64, 128, 128),
        "t1_large": (4, 32, 128, 128, 256, 256),
    }.items():
        x = torch.randn((N, C, IH, IW), device=DEV, dtype=torch.float16)
        outp = torch.empty((N, C, OH, OW), device=DEV, dtype=torch.float16)
        n_rows = N * C * OH
        rh = IH / OH
        s = x.stride()
        r = {}
        # host-side floors
        r["host_empty"] = bench(lambda: torch.empty((N, C, OH, OW), device=DEV,
                                                    dtype=torch.float16))
        r["host_touch_launch"] = bench(lambda: k_touch[(1,)](outp))
        r["host_full_run"] = bench(lambda: cand.run(x, [OH, OW], False))
        for br in (8, 16, 32):
            if n_rows % br:
                continue
            g = (n_rows // br,)
            for nm, kk, kw in [("flat", k_flat, {}), ("cube", k_cube, {}),
                               ("flat_nw1", k_flat, {"num_warps": 1}),
                               ("flat_nw8", k_flat, {"num_warps": 8})]:
                try:
                    r[f"{nm}_BR{br}"] = bench(
                        lambda kk=kk, g=g, br=br, kw=kw: kk[g](
                            outp, x, n_rows, s[0], s[1], s[2], rh,
                            C=C, IH=IH, IW=IW, OH=OH, OW=OW, BR=br, **kw))
                except Exception:
                    r[f"{nm}_BR{br}"] = "ERR:" + traceback.format_exc().splitlines()[-1][:120]
        out[tag] = r
    with open(os.path.join(ART, "store_host.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


main()
