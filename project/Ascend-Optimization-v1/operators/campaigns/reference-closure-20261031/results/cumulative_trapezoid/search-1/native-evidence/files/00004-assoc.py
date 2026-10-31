"""Diagnostic 2: which scan association reproduces torch.cumulative_trapezoid?

Variants (screening only; eval_round is authoritative):
  A) manual Hillis-Steele doubling with explicit per-step cast to the working dtype
  B) fully serial accumulation in the input dtype (lanes over the non-scan dims)
  C) tl.cumsum with different BLOCK widths (fp32 path)
"""
import importlib.util
import json
import os

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
    """Doubling scan; LOW=1 rounds every step to the input dtype."""
    N1: tl.constexpr = N - 1
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    yb = row * N
    ob = row * N1
    m = k < N1
    zero = tl.zeros((BLOCK,), Y.dtype.element_ty)
    carry = zero
    for t in tl.static_range(0, (N1 + BLOCK - 1) // BLOCK):
        off = t * BLOCK + k
        mm = off < N1
        a = tl.load(Y + yb + off, mask=mm, other=0.0).to(tl.float32)
        b = tl.load(Y + yb + off + 1, mask=mm, other=0.0).to(tl.float32)
        inc = ((a + b) * SCALE).to(Y.dtype.element_ty)
        s = inc if LOW else inc.to(tl.float32)
        for j in tl.static_range(0, LOG):
            step: tl.constexpr = 1 << j
            sh = tl.gather(s, tl.maximum(k - step, 0), 0)
            sh = tl.where(k >= step, sh, tl.zeros((BLOCK,), s.dtype))
            s = s + sh
            if LOW:
                s = s.to(Y.dtype.element_ty)
        if LOW:
            s = (s + carry).to(Y.dtype.element_ty)
            carry = tl.zeros((BLOCK,), Y.dtype.element_ty) + tl.sum(
                tl.where(k == BLOCK - 1, s, zero), 0)
        else:
            s = s + carry.to(tl.float32)
            carry = tl.zeros((BLOCK,), tl.float32) + tl.sum(
                tl.where(k == BLOCK - 1, s, tl.zeros((BLOCK,), tl.float32)), 0)
        tl.store(O + ob + off, s.to(O.dtype.element_ty), mask=mm)


@triton.jit
def _serial_rows(Y, O, nl, SCALE: tl.constexpr, N: tl.constexpr, N1: tl.constexpr,
                 RB: tl.constexpr):
    """[rows, N] contiguous: lanes over rows, serial over the last axis."""
    r = tl.arange(0, RB)
    ok = r >= 0
    carry = tl.zeros((RB,), Y.dtype.element_ty)
    for i in range(0, nl):
        a = tl.load(Y + r * N + i, mask=ok, other=0.0).to(tl.float32)
        b = tl.load(Y + r * N + i + 1, mask=ok, other=0.0).to(tl.float32)
        inc = ((a + b) * SCALE).to(Y.dtype.element_ty)
        carry = (carry + inc).to(Y.dtype.element_ty)
        tl.store(O + r * N1 + i, carry, mask=ok)


@triton.jit
def _serial_cols(Y, O, nl, SCALE: tl.constexpr, M: tl.constexpr, RB: tl.constexpr):
    """[N, M] contiguous, scan along axis 0: lanes over contiguous columns."""
    r = tl.arange(0, RB)
    ok = r < M
    carry = tl.zeros((RB,), Y.dtype.element_ty)
    for i in range(0, nl):
        a = tl.load(Y + i * M + r, mask=ok, other=0.0).to(tl.float32)
        b = tl.load(Y + (i + 1) * M + r, mask=ok, other=0.0).to(tl.float32)
        inc = ((a + b) * SCALE).to(Y.dtype.element_ty)
        carry = (carry + inc).to(Y.dtype.element_ty)
        tl.store(O + i * M + r, carry, mask=ok)


def build(shape, dt, seed=1729):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g).to(dtype=DT[dt], device='npu')


def stat(ref, got, rtol, atol):
    a, b = ref.float(), got.float()
    diff = (a - b).abs()
    bad = diff > (atol + rtol * a.abs())
    return {'nbad': int(bad.sum()), 'max': float(diff.max()),
            'bitexact': bool((a == b).all())}


def main():
    art = os.environ.get('KGS_DEBUG_ARTIFACTS', '.')
    rep = {}
    # 1) fp32 long axis: tl.cumsum with several BLOCK widths
    y = build([1024, 4096], 'float32')
    ref = torch.cumulative_trapezoid(y, dx=1, dim=-1)
    for blk in (4096, 1024):
        out = torch.empty([1024, 4095], dtype=y.dtype, device=y.device)
        cand._scan_last[(1024,)](y, out, 0.5, 4096, blk, 1)
        rep['t7_blockscan_%d' % blk] = stat(ref, out, 1e-4, 1e-4)
        print('t7 blk%d' % blk, rep['t7_blockscan_%d' % blk], flush=True)
    for low in (1, 0):
        out = torch.empty([1024, 4095], dtype=y.dtype, device=y.device)
        _hs[(1024,)](y, out, 0.5, 4096, 2048, 11, low)
        rep['t7_hs_%d' % low] = stat(ref, out, 1e-4, 1e-4)
        print('t7 hs low=%d' % low, rep['t7_hs_%d' % low], flush=True)
    del y, ref

    # 2) fp16 short rows
    y = build([128, 1024], 'float16')
    ref = torch.cumulative_trapezoid(y, dx=1, dim=-1)
    out = torch.empty([128, 1023], dtype=y.dtype, device=y.device)
    _hs[(128,)](y, out, 0.5, 1024, 1024, 10, 1)
    rep['t0_hs_low1'] = stat(ref, out, 0.015, 0.015)
    print('t0 hs low=1', rep['t0_hs_low1'], flush=True)
    out2 = torch.empty([128, 1023], dtype=y.dtype, device=y.device)
    _serial_rows[(1,)](y, out2, 1023, 0.5, 1024, 1023, 128)
    rep['t0_serial'] = stat(ref, out2, 0.015, 0.015)
    print('t0 serial', rep['t0_serial'], flush=True)
    del y, ref

    # 3) fp16 axis-0 (contiguous columns)
    y = build([1024, 128], 'float16')
    ref = torch.cumulative_trapezoid(y, dx=1, dim=0)
    out = torch.empty([1023, 128], dtype=y.dtype, device=y.device)
    _serial_cols[(1,)](y, out, 1023, 0.5, 128, 128)
    rep['t2_serial'] = stat(ref, out, 0.015, 0.015)
    print('t2 serial', rep['t2_serial'], flush=True)
    del y, ref

    # 4) bf16 long rows with low rounding sanity
    y = build([1024, 4096], 'bfloat16')
    ref = torch.cumulative_trapezoid(y, dx=1, dim=-1)
    out = torch.empty([1024, 4095], dtype=y.dtype, device=y.device)
    cand._scan_last[(1024,)](y, out, 0.5, 4096, 2048, 1)
    rep['t4_scanacc1'] = stat(ref, out, 0.015, 0.015)
    print('t4 scan acc1', rep['t4_scanacc1'], flush=True)

    # 5) fp16 long rows with doubling (tile = 2048)
    y = build([1024, 4096], 'float16')
    ref = torch.cumulative_trapezoid(y, dx=1, dim=-1)
    out = torch.empty([1024, 4095], dtype=y.dtype, device=y.device)
    _hs[(1024,)](y, out, 0.5, 4096, 2048, 11, 1)
    rep['t1_hs_low1'] = stat(ref, out, 0.015, 0.015)
    print('t1 hs low=1', rep['t1_hs_low1'], flush=True)

    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, 'assoc.json'), 'w') as fh:
        json.dump(rep, fh, indent=1)


main()
