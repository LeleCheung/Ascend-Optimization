"""Diagnostic 4: doubling-scan tiling/carry for long axes + axis-0 variant, with timing."""
import importlib.util
import json
import os
import time

import torch
import triton
import triton.language as tl

spec = importlib.util.spec_from_file_location('cand', 'tmp/main.py')
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)

DT = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}


@triton.jit
def _hs(Y, O, SCALE: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr,
        LOG: tl.constexpr, LOW: tl.constexpr):
    N1: tl.constexpr = N - 1
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    yb = row * N
    ob = row * N1
    z16 = tl.zeros((BLOCK,), Y.dtype.element_ty)
    z32 = tl.zeros((BLOCK,), tl.float32)
    c16 = z16
    c32 = z32
    for t in tl.static_range(0, (N1 + BLOCK - 1) // BLOCK):
        off = t * BLOCK + k
        mm = off < N1
        a = tl.load(Y + yb + off, mask=mm, other=0.0).to(tl.float32)
        b = tl.load(Y + yb + off + 1, mask=mm, other=0.0).to(tl.float32)
        inc = ((a + b) * SCALE).to(Y.dtype.element_ty)
        if LOW:
            s = inc
        else:
            s = inc.to(tl.float32)
        for j in tl.static_range(0, LOG):
            sh = tl.gather(s, tl.maximum(k - (1 << j), 0), 0)
            if LOW:
                sh = tl.where(k >= (1 << j), sh, z16)
                s = (s + sh).to(Y.dtype.element_ty)
            else:
                sh = tl.where(k >= (1 << j), sh, z32)
                s = s + sh
        if LOW:
            s = (s + c16).to(Y.dtype.element_ty)
            c16 = z16 + tl.sum(tl.where(k == BLOCK - 1, s, z16), 0)
        else:
            s = s + c32
            c32 = z32 + tl.sum(tl.where(k == BLOCK - 1, s, z32), 0)
        tl.store(O + ob + off, s.to(O.dtype.element_ty), mask=mm)


@triton.jit
def _hs0(Y, O, SCALE: tl.constexpr, N: tl.constexpr, M: tl.constexpr,
         KB: tl.constexpr, CBW: tl.constexpr, LOG: tl.constexpr, LOW: tl.constexpr):
    """axis-0 doubling scan (contiguous inner), one full-length tile."""
    N1: tl.constexpr = N - 1
    c = tl.program_id(0)
    rows = tl.arange(0, KB)
    cols = c * CBW + tl.arange(0, CBW)
    cmask = cols < M
    mask = (rows < N1)[:, None] & cmask[None, :]
    idx = rows[:, None] * M + cols[None, :]
    a = tl.load(Y + idx, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(Y + idx + M, mask=mask, other=0.0).to(tl.float32)
    inc = ((a + b) * SCALE).to(Y.dtype.element_ty)
    if LOW:
        s = inc
    else:
        s = inc.to(tl.float32)
    z = tl.zeros((KB, CBW), Y.dtype.element_ty)
    sh_idx = tl.broadcast_to(rows[:, None], (KB, CBW))
    for j in tl.static_range(0, LOG):
        g = tl.gather(s, tl.maximum(sh_idx - (1 << j), 0), 0)
        if LOW:
            g = tl.where((rows >= (1 << j))[:, None], g, z)
            s = (s + g).to(Y.dtype.element_ty)
        else:
            g = tl.where((rows >= (1 << j))[:, None], g, z.to(tl.float32))
            s = s + g
    tl.store(O + idx, s.to(O.dtype.element_ty), mask=mask)


def build(shape, dt, seed=1729):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g).to(dtype=DT[dt], device='npu')


def stat(ref, got, rtol, atol):
    a, b = ref.float(), got.float()
    diff = (a - b).abs()
    bad = diff > (atol + rtol * a.abs())
    return {'nbad': int(bad.sum()), 'max': float(diff.max()),
            'bitexact': bool((a == b).all())}


def bench(fn, iters=20):
    fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def main():
    art = os.environ.get('KGS_DEBUG_ARTIFACTS', '.')
    rep = {}
    t16 = (0.015, 0.015)
    t32 = (1e-4, 1e-4)

    # fp16 long rows
    y = build([1024, 4096], 'float16')
    ref = torch.cumulative_trapezoid(y, dx=1, dim=-1)
    for blk, log in ((4096, 12), (2048, 11)):
        out = torch.empty([1024, 4095], dtype=y.dtype, device=y.device)
        _hs[(1024,)](y, out, 0.5, 4096, blk, log, 1)
        key = 't1_fp16_hs_blk%d' % blk
        rep[key] = stat(ref, out, *t16)
        rep[key]['ms'] = bench(lambda: _hs[(1024,)](y, out, 0.5, 4096, blk, log, 1), 10)
        print(key, rep[key], flush=True)
    rep['t1_ms_ref'] = bench(lambda: torch.cumulative_trapezoid(y, dx=1, dim=-1), 10)
    print('t1 ref ms', rep['t1_ms_ref'], flush=True)
    del y, ref

    # fp32 long rows
    y = build([1024, 4096], 'float32')
    ref = torch.cumulative_trapezoid(y, dx=1, dim=-1)
    for blk, log in ((4096, 12), (2048, 11)):
        out = torch.empty([1024, 4095], dtype=y.dtype, device=y.device)
        _hs[(1024,)](y, out, 0.5, 4096, blk, log, 0)
        key = 't7_fp32_hs_blk%d' % blk
        rep[key] = stat(ref, out, *t32)
        rep[key]['ms'] = bench(lambda: _hs[(1024,)](y, out, 0.5, 4096, blk, log, 0), 10)
        print(key, rep[key], flush=True)
    del y, ref

    # fp16 axis 0, full-length single tile
    y = build([1024, 128], 'float16')
    ref = torch.cumulative_trapezoid(y, dx=1, dim=0)
    for cbw in (16, 32):
        try:
            out = torch.empty([1023, 128], dtype=y.dtype, device=y.device)
            _hs0[(128 // cbw,)](y, out, 0.5, 1024, 128, 1024, cbw, 10, 1)
            key = 't2_fp16_hs0_cbw%d' % cbw
            rep[key] = stat(ref, out, *t16)
            rep[key]['ms'] = bench(
                lambda: _hs0[(128 // cbw,)](y, out, 0.5, 1024, 128, 1024, cbw, 10, 1), 10)
            print(key, rep[key], flush=True)
        except Exception as exc:  # noqa: BLE001
            print('t2 cbw%d failed: %s' % (cbw, exc), flush=True)
    rep['t2_ms_ref'] = bench(lambda: torch.cumulative_trapezoid(y, dx=1, dim=0), 10)
    print('t2 ref ms', rep['t2_ms_ref'], flush=True)

    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, 'assoc3.json'), 'w') as fh:
        json.dump(rep, fh, indent=1)


main()
