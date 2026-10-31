"""Device-bound kernel variants for 4096x4096: masked vs unmasked, BLOCK, math dtype."""
import json
import os
import sys
import time

import torch
import torch_npu
import triton
import triton.language as tl

sys.path.insert(0, 'tmp')

ART = os.environ.get('KGS_DEBUG_ARTIFACTS', '.')
DEV = 'npu:0'
res = {}


def dump():
    with open(os.path.join(ART, 'kvar.json'), 'w') as fh:
        json.dump(res, fh, indent=1, sort_keys=True)


def timed(fn, reps=300, warmup=60):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    t2 = time.perf_counter()
    return {'submit_us': round((t1 - t0) / reps * 1e6, 1),
            'total_us': round((t2 - t0) / reps * 1e6, 1)}


@triton.jit
def _masked(A, B, N: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    a = tl.load(A + off, m, other=0)
    b = tl.load(B + off, m, other=1)
    x = a.to(tl.float32)
    y = b.to(tl.float32)
    v = tl.where(y != y, float('nan'), tl.where(x == 0.0, 0.0, x * tl.log(y)))
    tl.store(A + off, v, m)


@triton.jit
def _unmasked(A, B, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + off)
    b = tl.load(B + off)
    x = a.to(tl.float32)
    y = b.to(tl.float32)
    v = tl.where(y != y, float('nan'), tl.where(x == 0.0, 0.0, x * tl.log(y)))
    tl.store(A + off, v)


@triton.jit
def _unmasked_native(A, B, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + off)
    b = tl.load(B + off)
    v = tl.where(b != b, float('nan'), tl.where(a == 0.0, 0.0, a * tl.log(b)))
    tl.store(A + off, v)


def mk(n, dt):
    a = (torch.randn(n, device=DEV, dtype=dt).abs() + 0.05)
    b = (torch.randn(n, device=DEV, dtype=dt).abs() + 0.05)
    return a, b


BIG = 4096 * 4096
res['variants'] = {}
for name, dt in (('f16', torch.float16), ('f32', torch.float32)):
    X, Y = mk(BIG, dt)
    row = {}
    row['torch_xlogy_'] = timed(lambda: X.xlogy_(Y))
    for blk in (4096, 8192):
        g = (BIG + blk - 1) // blk
        try:
            row['masked_%d' % blk] = timed(lambda blk=blk, g=g: _masked[(g,)](X, Y, BIG, blk))
        except Exception as exc:  # noqa: BLE001
            row['masked_%d' % blk] = {'error': repr(exc)[:160]}
        try:
            row['unmasked_%d' % blk] = timed(lambda blk=blk, g=g: _unmasked[(g,)](X, Y, blk))
        except Exception as exc:  # noqa: BLE001
            row['unmasked_%d' % blk] = {'error': repr(exc)[:160]}
    try:
        row['unmasked_native_4096'] = timed(
            lambda: _unmasked_native[(BIG // 4096,)](X, Y, 4096))
    except Exception as exc:  # noqa: BLE001
        row['unmasked_native_4096'] = {'error': repr(exc)[:160]}
    res['variants'][name] = row
    dump()

print(json.dumps(res, indent=1, sort_keys=True))
