"""Diagnostic 7: host-launch overhead reduction and corrected transpose path for axis-0."""
import json
import os
import time

import torch
import triton
import triton.language as tl
import torch_npu  # noqa: F401

_RAW = {}


def _raw_stream(idx):
    getter = _RAW.get(idx)
    if getter is None:
        raw = getattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait', None)
        expected = int(torch.npu.current_stream(idx).npu_stream)
        if raw is not None:
            try:
                if int(raw(idx)) == expected:
                    getter = lambda: raw(idx)  # noqa: E731
            except TypeError:
                if int(raw()) == expected:
                    getter = raw
        if getter is None:
            from triton.runtime import driver
            getter = lambda: driver.active.get_current_stream(idx)  # noqa: E731
        _RAW[idx] = getter
    return getter()


class Fast:
    def __init__(self, kern):
        self.kern = kern
        self.entries = {}

    def __getitem__(self, grid):
        def invoke(*args):
            key = (tuple(grid),) + tuple(
                (a.data_ptr() & 15, tuple(a.shape), tuple(a.stride()), a.dtype)
                if torch.is_tensor(a) else a for a in args)
            e = self.entries.get(key)
            if e is None:
                compiled = self.kern[grid](*args)
                launcher = compiled.run
                shape = tuple(grid) + (1,) * (3 - len(grid))
                self.entries[key] = (launcher.launch, compiled.function,
                                     compiled.packed_metadata, shape)
                return
            ptrs = tuple(a.data_ptr() if torch.is_tensor(a) else None for a in args)
            idx = args[0].device.index or 0
            e[0](*e[3], _raw_stream(idx), e[1], e[2], None, None, None, *ptrs)
        return invoke


@triton.jit
def _scan_vec(Y, O, SCALE: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr,
              LOG: tl.constexpr, LOW: tl.constexpr):
    N1: tl.constexpr = N - 1
    acc16 = tl.zeros((BLOCK,), Y.dtype.element_ty)
    acc32 = tl.zeros((BLOCK,), tl.float32)
    k = tl.arange(0, BLOCK)
    row = tl.program_id(0)
    yb = row * N
    ob = row * N1
    mm = k < N1
    a = tl.load(Y + yb + k, mask=mm, other=0.0).to(tl.float32)
    b = tl.load(Y + yb + k + 1, mask=mm, other=0.0).to(tl.float32)
    inc = ((a + b) * SCALE).to(Y.dtype.element_ty)
    s = inc if LOW else inc.to(tl.float32)
    for j in tl.static_range(0, LOG):
        g = tl.gather(s, tl.maximum(k - (1 << j), 0), 0)
        if LOW:
            s = (s + tl.where(k >= (1 << j), g, acc16)).to(Y.dtype.element_ty)
        else:
            s = s + tl.where(k >= (1 << j), g, acc32)
    tl.store(O + ob + k, s.to(O.dtype.element_ty), mask=mm)


@triton.jit
def _hs_scratch(Y, O, SCR, SCALE: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr,
                LOG: tl.constexpr, LOW: tl.constexpr):
    N1: tl.constexpr = N - 1
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    yb = row * N
    ob = row * N1
    sb = row * BLOCK
    mm = k < N1
    a = tl.load(Y + yb + k, mask=mm, other=0.0).to(tl.float32)
    b = tl.load(Y + yb + k + 1, mask=mm, other=0.0).to(tl.float32)
    inc = ((a + b) * SCALE).to(Y.dtype.element_ty)
    s = inc if LOW else inc.to(tl.float32)
    for j in tl.static_range(0, LOG):
        tl.store(SCR + sb + k, s)
        tl.debug_barrier()
        g = tl.load(SCR + sb + k - (1 << j), mask=k >= (1 << j), other=0.0)
        if LOW:
            s = (s + g).to(Y.dtype.element_ty)
        else:
            s = s + g
    tl.store(O + ob + k, s.to(O.dtype.element_ty), mask=mm)


@triton.jit
def _transpose2d(SRC, DST, R: tl.constexpr, C: tl.constexpr, SA: tl.constexpr,
                 DA: tl.constexpr, RB: tl.constexpr, CB: tl.constexpr):
    r = tl.program_id(0) * RB + tl.arange(0, RB)
    c = tl.program_id(1) * CB + tl.arange(0, CB)
    rm = r < R
    cm = c < C
    t = tl.load(SRC + r[:, None] * SA + c[None, :], mask=rm[:, None] & cm[None, :],
                other=0.0)
    tl.store(DST + c[:, None] * DA + r[None, :], tl.trans(t),
             mask=cm[:, None] & rm[None, :])


FAST_VEC = Fast(_scan_vec)
FAST_SCR = Fast(_hs_scratch)
FAST_TR = Fast(_transpose2d)


def bench(fn, iters=50):
    fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def stat(ref, got, rtol, atol):
    a, b = ref.float(), got.float()
    diff = (a - b).abs()
    bad = diff > (atol + rtol * a.abs())
    return {'nbad': int(bad.sum()), 'max': float(diff.max())}


def main():
    art = os.environ.get('KGS_DEBUG_ARTIFACTS', '.')
    rep = {}
    g = torch.Generator().manual_seed(1729)
    # A) plain vs fast launch, gather, t0
    y = torch.randn([128, 1024], generator=g).to(dtype=torch.float16, device='npu')
    out = torch.empty(128, 1023, dtype=torch.float16, device='npu')
    rep['t0_plain'] = bench(lambda: _scan_vec[(128,)](y, out, 0.5, 1024, 1024, 10, 1))
    rep['t0_fast'] = bench(lambda: FAST_VEC[(128,)](y, out, 0.5, 1024, 1024, 10, 1))
    print('A', rep['t0_plain'], rep['t0_fast'], flush=True)
    ref = torch.cumulative_trapezoid(y, dx=1, dim=-1)
    FAST_VEC[(128,)](y, out, 0.5, 1024, 1024, 10, 1)
    torch.npu.synchronize()
    rep['t0_num'] = stat(ref, out, 0.015, 0.015)
    print('A num', rep['t0_num'], flush=True)
    del y, out, ref

    # B) scratch doubling, t1, plain vs fast
    y = torch.randn([1024, 4096], generator=g).to(dtype=torch.float16, device='npu')
    out = torch.empty(1024, 4095, dtype=torch.float16, device='npu')
    scr = torch.empty(1024 * 4096, dtype=torch.float16, device='npu')
    rep['t1_plain'] = bench(lambda: _hs_scratch[(1024,)](y, out, scr, 0.5, 4096, 4096, 12, 1), 20)
    rep['t1_fast'] = bench(lambda: FAST_SCR[(1024,)](y, out, scr, 0.5, 4096, 4096, 12, 1), 20)
    print('B', rep['t1_plain'], rep['t1_fast'], flush=True)
    ref = torch.cumulative_trapezoid(y, dx=1, dim=-1)
    FAST_SCR[(1024,)](y, out, scr, 0.5, 4096, 4096, 12, 1)
    torch.npu.synchronize()
    rep['t1_num'] = stat(ref, out, 0.015, 0.015)
    print('B num', rep['t1_num'], flush=True)
    del y, out, ref, scr

    # C) dim0 via transpose + vec + transpose, corrected grid
    for dtname, dt in (('f16', torch.float16), ('f32', torch.float32)):
        y = torch.randn([1024, 128], generator=g).to(dtype=dt, device='npu')
        ref = torch.cumulative_trapezoid(y, dx=1, dim=0)
        rtol, atol = (1e-4, 1e-4) if dt == torch.float32 else (0.015, 0.015)
        yt = torch.empty(128, 1024, dtype=dt, device='npu')
        ot = torch.empty(128, 1023, dtype=dt, device='npu')
        out = torch.empty(1023, 128, dtype=dt, device='npu')
        low = 0 if dt == torch.float32 else 1

        def run_plain():
            _transpose2d[(16, 2)](y, yt, 1024, 128, 128, 1024, 64, 64)
            _scan_vec[(128,)](yt, ot, 0.5, 1024, 1024, 10, low)
            _transpose2d[(2, 16)](ot, out, 128, 1023, 1023, 128, 64, 64)

        def run_fast():
            FAST_TR[(16, 2)](y, yt, 1024, 128, 128, 1024, 64, 64)
            FAST_VEC[(128,)](yt, ot, 0.5, 1024, 1024, 10, low)
            FAST_TR[(2, 16)](ot, out, 128, 1023, 1023, 128, 64, 64)

        for tag, fn in (('plain', run_plain), ('fast', run_fast)):
            try:
                fn()
                torch.npu.synchronize()
                rep['%s_tr_%s' % (dtname, tag)] = {
                    'ms': bench(fn, 20), 'num': stat(ref, out, rtol, atol)}
                print('C', dtname, tag, rep['%s_tr_%s' % (dtname, tag)], flush=True)
            except Exception as exc:  # noqa: BLE001
                print('C', dtname, tag, 'fail', str(exc)[:300], flush=True)
        rep['%s_empty_ms' % dtname] = bench(
            lambda: torch.empty(1023, 128, dtype=dt, device='npu'), 100)
        print('C', dtname, 'empty', rep['%s_empty_ms' % dtname], flush=True)
        del y, ref, yt, ot, out

    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, 'perf7.json'), 'w') as fh:
        json.dump(rep, fh, indent=1)


main()
