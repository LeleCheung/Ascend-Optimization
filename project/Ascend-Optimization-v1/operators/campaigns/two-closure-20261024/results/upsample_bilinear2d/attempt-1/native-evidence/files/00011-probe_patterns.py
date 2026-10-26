"""Bounded target-side roofline probe for upsample_bilinear2d.

Measures, on the real device:
  * pure 1D contiguous copy bandwidth (the floor for the large geometry)
  * the penalty of a 1-element-offset (misaligned) contiguous load
  * an elementwise 3-load vertical-combine pass (contiguous loads, strided rows)
  * the fused variant, adding in-register tl.gather + tl.join + full-row store
  * the same fused variant using shifted GM loads instead of gather
  * host-side decomposition of the current run() (empty / launch / full call)
All numbers are diagnostic; the authoritative verdict stays with eval_round.
"""

import json
import os
import sys
import time

import torch
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
RES = {}


def rec(k, v):
    RES[k] = v
    try:
        with open(os.path.join(ART, "probe.json"), "w") as f:
            json.dump(RES, f, indent=1, default=str)
    except Exception as exc:  # never let artifact IO kill the probe
        print("artifact io error", exc, flush=True)
    print("RESULT", k, v, flush=True)


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------
@triton.jit
def k_copy(o, i, base_o, base_i, SHIFT: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(i + base_i + SHIFT + off)
    tl.store(o + base_o + off, v)


@triton.jit
def k_min(o, i, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(o + off, tl.load(i + off))


@triton.jit
def k_v3(o, i, PAD: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
         ROWS: tl.constexpr, NBLK: tl.constexpr):
    """Vertical 2x combine only: 3 fully contiguous 1D loads, strided-row store."""
    pid = tl.program_id(0)
    img = pid // NBLK
    blk = pid % NBLK
    b0 = PAD + img * (IH * IW) + blk * (ROWS * IW)
    off = tl.arange(0, ROWS * IW)
    a = tl.load(i + b0 - IW + off).to(tl.float32)
    b = tl.load(i + b0 + off).to(tl.float32)
    c = tl.load(i + b0 + IW + off).to(tl.float32)
    e = 0.25 * a + 0.75 * b
    d = 0.75 * b + 0.25 * c
    rows = blk * ROWS + tl.arange(0, ROWS)
    ob = img * (2 * IH * 2 * IW)
    offs = ob + (rows * 2)[:, None] * (2 * IW) + tl.arange(0, IW)[None, :]
    tl.store(o + offs, tl.reshape(e, (ROWS, IW)).to(o.dtype.element_ty))
    tl.store(o + offs + 2 * IW, tl.reshape(d, (ROWS, IW)).to(o.dtype.element_ty))


@triton.jit
def k_fused(o, i, PAD: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
            ROWS: tl.constexpr, NBLK: tl.constexpr, GATHER: tl.constexpr):
    """Vertical combine + horizontal 2x expand + full contiguous output rows."""
    pid = tl.program_id(0)
    img = pid // NBLK
    blk = pid % NBLK
    b0 = PAD + img * (IH * IW) + blk * (ROWS * IW)
    off = tl.arange(0, ROWS * IW)
    a = tl.load(i + b0 - IW + off).to(tl.float32)
    b = tl.load(i + b0 + off).to(tl.float32)
    c = tl.load(i + b0 + IW + off).to(tl.float32)
    e = tl.reshape(0.25 * a + 0.75 * b, (ROWS, IW))
    d = tl.reshape(0.75 * b + 0.25 * c, (ROWS, IW))

    k = tl.arange(0, IW)[None, :]
    il = tl.broadcast_to(tl.maximum(k - 1, 0), (ROWS, IW))
    ir = tl.broadcast_to(tl.minimum(k + 1, IW - 1), (ROWS, IW))
    if GATHER:
        em = tl.gather(e, il, 1)
        ep = tl.gather(e, ir, 1)
        dm = tl.gather(d, il, 1)
        dp = tl.gather(d, ir, 1)
    else:
        # shifted GM loads instead of in-register gather (misaligned bases)
        am = tl.reshape(tl.load(i + b0 - IW - 1 + off).to(tl.float32), (ROWS, IW))
        bm = tl.reshape(tl.load(i + b0 - 1 + off).to(tl.float32), (ROWS, IW))
        cm = tl.reshape(tl.load(i + b0 + IW - 1 + off).to(tl.float32), (ROWS, IW))
        ap = tl.reshape(tl.load(i + b0 - IW + 1 + off).to(tl.float32), (ROWS, IW))
        bp = tl.reshape(tl.load(i + b0 + 1 + off).to(tl.float32), (ROWS, IW))
        cp = tl.reshape(tl.load(i + b0 + IW + 1 + off).to(tl.float32), (ROWS, IW))
        em = 0.25 * am + 0.75 * bm
        ep = 0.25 * ap + 0.75 * bp
        dm = 0.75 * bm + 0.25 * cm
        dp = 0.75 * bp + 0.25 * cp

    he = tl.join(0.25 * em + 0.75 * e, 0.75 * e + 0.25 * ep)
    hd = tl.join(0.25 * dm + 0.75 * d, 0.75 * d + 0.25 * dp)

    rows = blk * ROWS + tl.arange(0, ROWS)
    ob = img * (2 * IH * 2 * IW)
    j = tl.arange(0, 2)
    offs = (ob + (rows * 2)[:, None, None] * (2 * IW)
            + tl.arange(0, IW)[None, :, None] * 2 + j[None, None, :])
    tl.store(o + offs, he.to(o.dtype.element_ty))
    tl.store(o + offs + 2 * IW, hd.to(o.dtype.element_ty))


# --------------------------------------------------------------------------
# harness helpers
# --------------------------------------------------------------------------
def dev_ms(fn, K=50, reps=3):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    best = float("inf")
    for _ in range(reps):
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        for _ in range(K):
            fn()
        e.record()
        torch.npu.synchronize()
        best = min(best, s.elapsed_time(e) / K)
    return best


def wall_ms(fn, K=100, sync_each=True):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    if sync_each:
        for _ in range(K):
            fn()
            torch.npu.synchronize()
    else:
        for _ in range(K):
            fn()
        torch.npu.synchronize()
    return (time.perf_counter() - t0) / K * 1e3


def patterns():
    dev = torch.device("npu:0")
    NC, IH, IW = 128, 128, 128
    OH, OW = 2 * IH, 2 * IW
    PAD = IW
    n_in = NC * IH * IW
    n_out = NC * OH * OW
    BLK = 8192

    for dt in (torch.float32, torch.float16):
        tag = "f32" if dt is torch.float32 else "f16"
        ibuf = torch.randn(n_in + 2 * PAD, device=dev, dtype=dt)
        obuf = torch.empty(n_out, device=dev, dtype=dt)

        gref = (n_out // BLK,)
        rec(f"copy_fwd_{tag}_ms",
            dev_ms(lambda: k_copy[gref](obuf, obuf, 0, 0, SHIFT=0, BLOCK=BLK)))
        rec(f"copy_shift1_{tag}_ms",
            dev_ms(lambda: k_copy[gref](obuf, ibuf, 0, PAD, SHIFT=1, BLOCK=BLK)))
        fin = (n_in // BLK,)
        rec(f"copy_invol_{tag}_ms",
            dev_ms(lambda: k_copy[fin](obuf, ibuf, 0, PAD, SHIFT=0, BLOCK=BLK)))

        rowses = (8, 16)
        for ROWS in rowses:
            nblk = IH // ROWS
            grid = (NC * nblk,)
            g3 = lambda ROWS=ROWS, nblk=nblk, grid=grid: k_v3[grid](
                obuf, ibuf, PAD=PAD, IH=IH, IW=IW, ROWS=ROWS, NBLK=nblk)
            rec(f"v3_r{ROWS}_{tag}_ms", dev_ms(g3))
            gf = lambda ROWS=ROWS, nblk=nblk, grid=grid: k_fused[grid](
                obuf, ibuf, PAD=PAD, IH=IH, IW=IW, ROWS=ROWS, NBLK=nblk, GATHER=True)
            rec(f"fused_r{ROWS}_{tag}_ms", dev_ms(gf))

        if dt is torch.float32:
            ROWS = 8
            nblk = IH // ROWS
            grid = (NC * nblk,)
            gl = lambda: k_fused[grid](obuf, ibuf, PAD=PAD, IH=IH, IW=IW, ROWS=ROWS,
                                       NBLK=nblk, GATHER=False)
            rec(f"fusedload_r{ROWS}_{tag}_ms", dev_ms(gl))

            x = ibuf[PAD:PAD + n_in].view(1, NC, IH, IW)
            ref = torch.nn.functional.interpolate(
                x, size=(OH, OW), mode="bilinear",
                align_corners=False).reshape(-1)
            got = obuf
            worst = 0.0
            for img in range(NC):
                for blk in range(1, nblk - 1):
                    o0 = img * (OH * OW) + blk * ROWS * 2 * OW
                    o1 = o0 + ROWS * 2 * OW
                    d = (got[o0:o1] - ref[o0:o1]).abs()
                    worst = max(worst, float(d.max()))
            rec(f"fused_numeric_maxdiff_interior_{tag}", worst)
        del ibuf, obuf


def main():
    dev = torch.device("npu:0")
    torch.npu.set_device(dev)
    try:
        props = torch.npu.get_device_properties(0)
        rec("device", {"name": props.name,
                       "num_sm": getattr(props, "num_sm", None),
                       "num_vectorcore": getattr(props, "num_vectorcore", None),
                       "num_aicore": getattr(props, "num_aicore", None)})
    except Exception as exc:
        rec("device", str(exc))
    rec("torch", torch.__version__)
    rec("triton", triton.__version__)
    rec("cwd", os.getcwd())
    rec("artifacts", ART)

    try:
        patterns()
    except Exception as exc:
        import traceback
        rec("pattern_error", traceback.format_exc())

    # ---------------- host-side decomposition (small geometry) ----------------
    try:
        ws = os.environ.get("PROBE_WORKSPACE", os.getcwd())
        sys.path.insert(0, ws)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "cand", os.path.join(ws, "tmp", "main.py"))
        cand = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cand)

        xs = torch.randn(1, 16, 64, 64, device=dev, dtype=torch.float16)
        xl = torch.randn(4, 32, 128, 128, device=dev, dtype=torch.float32)
        os_small = (128, 128)
        os_large = (256, 256)

        aten = torch.ops.aten.upsample_bilinear2d
        rec("aten_small_wall_sync_ms", wall_ms(lambda: aten(xs, os_small, False, None, None)))
        rec("aten_large_wall_sync_ms", wall_ms(lambda: aten(xl, os_large, False, None, None), K=50))
        rec("aten_small_dev_ms", dev_ms(lambda: aten(xs, os_small, False, None, None)))
        rec("aten_large_dev_ms", dev_ms(lambda: aten(xl, os_large, False, None, None), K=30))

        rec("cand_small_wall_sync_ms", wall_ms(lambda: cand.run(xs, os_small, False, None, None)))
        rec("cand_large_wall_sync_ms", wall_ms(lambda: cand.run(xl, os_large, False, None, None), K=50))
        rec("cand_small_dev_ms", dev_ms(lambda: cand.run(xs, os_small, False, None, None)))
        rec("cand_large_dev_ms", dev_ms(lambda: cand.run(xl, os_large, False, None, None), K=30))

        rec("empty_small_wall_sync_ms",
            wall_ms(lambda: torch.empty((1, 16, 128, 128), device=dev, dtype=torch.float16)))
        rec("empty_large_wall_sync_ms",
            wall_ms(lambda: torch.empty((4, 32, 256, 256), device=dev, dtype=torch.float32)))

        oo = torch.empty(8192, device=dev, dtype=torch.float16)
        ii = torch.empty(8192, device=dev, dtype=torch.float16)
        rec("minlaunch_wall_sync_ms", wall_ms(lambda: k_min[(1,)](oo, ii, BLOCK=8192)))
        rec("minlaunch_dev_ms", dev_ms(lambda: k_min[(1,)](oo, ii, BLOCK=8192)))

        key = (1, 16, 64, 64, 128, 128, False, None, None, xs.stride(), xs.dtype)
        ent = cand._LAUNCH_CACHE.get(key)
        if ent is None:
            cand.run(xs, os_small, False, None, None)
            ent = cand._LAUNCH_CACHE[key]
        outs = torch.empty((1, 16, 128, 128), device=dev, dtype=torch.float16)
        n_rows, s_n, s_c, s_h, rh, kc, kih, kiw, koh, kow, br, iwp = ent[2]
        kw = dict(C=kc, IH=kih, IW=kiw, OH=koh, OW=kow, BR=br, IWP=iwp)
        rec("cand_kernel_ent", [br, iwp, n_rows, list(ent[1])])
        rec("cand_kernellaunch_wall_sync_ms",
            wall_ms(lambda: cand._bilinear_int_kernel[ent[1]](
                outs, xs, n_rows, s_n, s_c, s_h, rh, **kw)))
        rec("cand_kernellaunch_dev_ms",
            dev_ms(lambda: cand._bilinear_int_kernel[ent[1]](
                outs, xs, n_rows, s_n, s_c, s_h, rh, **kw)))
    except Exception as exc:
        import traceback
        rec("host_error", traceback.format_exc())

    rec("done", True)


if __name__ == "__main__":
    main()
