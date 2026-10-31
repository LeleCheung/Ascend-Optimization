"""Diagnostic 3: serial vs doubling scan association against torch.cumulative_trapezoid."""
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
    N1: tl.constexpr = N - 1
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    yb = row * N
    ob = row * N1
    z16 = tl.zeros((BLOCK,), Y.dtype.element_ty)
    z32 = tl.zeros((BLOCK,), tl.float32)
    carry16 = z16
    carry32 = z32
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
            s = (s + carry16).to(Y.dtype.element_ty)
            carry16 = z16 + tl.sum(tl.where(k == BLOCK - 1, s, z16), 0)
        else:
            s = s + carry32
            carry32 = z32 + tl.sum(tl.where(k == BLOCK - 1, s, z32), 0)
        tl.store(O + ob + off, s.to(O.dtype.element_ty), mask=mm)


@triton.jit
def _serial_rows(Y, O, nl, SCALE: tl.constexpr, N: tl.constexpr, N1: tl.constexpr,
                 RB: tl.constexpr):
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
    t16 = (0.015, 0.015)
    t32 = (1e-4, 1e-4)

    # A) fp16 [128,1024] dim=-1
    y = build([128, 1024], 'float16')
    ref = torch.cumulative_trapezoid(y, dx=1, dim=-1)
    out = torch.empty([128, 1023], dtype=y.dtype, device=y.device)
    _serial_rows[(1,)](y, out, 1023, 0.5, 1024, 1023, 128)
    rep['t0_serial'] = stat(ref, out, *t16)
    print('t0 serial', rep['t0_serial'], flush=True)
    out = torch.empty([128, 1023], dtype=y.dtype, device=y.device)
    _hs[(128,)](y, out, 0.5, 1024, 1024, 10, 1)
    rep['t0_hs16'] = stat(ref, out, *t16)
    print('t0 hs16', rep['t0_hs16'], flush=True)
    del y, ref

    # B) fp16 [1024,128] dim=0
    y = build([1024, 128], 'float16')
    ref = torch.cumulative_trapezoid(y, dx=1, dim=0)
    out = torch.empty([1023, 128], dtype=y.dtype, device=y.device)
    _serial_cols[(1,)](y, out, 1023, 0.5, 128, 128)
    rep['t2_serial'] = stat(ref, out, *t16)
    print('t2 serial', rep['t2_serial'], flush=True)
    del y, ref

    # C) fp32 [128,1024] dim=-1 serial
    y = build([128, 1024], 'float32')
    ref = torch.cumulative_trapezoid(y, dx=1, dim=-1)
    out = torch.empty([128, 1023], dtype=y.dtype, device=y.device)
    _serial_rows[(1,)](y, out, 1023, 0.5, 1024, 1023, 128)
    rep['t6_serial'] = stat(ref, out, *t32)
    print('t6 serial', rep['t6_serial'], flush=True)
    del y, ref

    # D) bf16 [1024,4096] dim=-1: doubling with fp16-style low rounding must NOT be used
    y = build([1024, 4096], 'bfloat16')
    ref = torch.cumulative_trapezoid(y, dx=1, dim=-1)
    out = torch.empty([1024, 4095], dtype=y.dtype, device=y.device)
    _hs[(1024,)](y, out, 0.5, 4096, 2048, 11, 1)
    rep['t4_hslow'] = stat(ref, out, *t16)
    print('t4 hs low', rep['t4_hslow'], flush=True)
    del y, ref

    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, 'assoc2.json'), 'w') as fh:
        json.dump(rep, fh, indent=1)


main()
