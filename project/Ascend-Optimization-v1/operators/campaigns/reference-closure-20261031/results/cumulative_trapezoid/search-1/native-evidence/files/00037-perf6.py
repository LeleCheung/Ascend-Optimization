"""Diagnostic 6: cost breakdown + alternatives for slow paths.

A) launch overhead of a trivial kernel
B) last-axis doubling: tl.gather vs global scratch ping-pong, num_warps sweep
C) axis-0 (dim0) path: current per-column, transpose+vec+transpose, chunked carry
"""
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


@triton.jit
def _trivial(X):
    tl.store(X + tl.program_id(0), 1.0)


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


@triton.jit
def _dim0_tile(Y, O, SCALE: tl.constexpr, N: tl.constexpr, M: tl.constexpr,
               KB: tl.constexpr, LOG: tl.constexpr, CBW: tl.constexpr,
               LOW: tl.constexpr, TILES: tl.constexpr):
    N1: tl.constexpr = N - 1
    kb = tl.arange(0, KB)
    kbi = tl.broadcast_to(kb[:, None], (KB, CBW))
    col = tl.program_id(0) * CBW + tl.arange(0, CBW)
    ok = col < M
    z16 = tl.zeros((KB, CBW), Y.dtype.element_ty)
    z32 = tl.zeros((KB, CBW), tl.float32)
    c16 = tl.zeros((CBW,), Y.dtype.element_ty)
    c32 = tl.zeros((CBW,), tl.float32)
    carry = c16 if LOW else c32
    for t in tl.static_range(0, TILES):
        rows = t * KB + kb
        mm = (rows < N1)[:, None] & ok[None, :]
        idx = rows[:, None] * M + col[None, :]
        a = tl.load(Y + idx, mask=mm, other=0.0).to(tl.float32)
        b = tl.load(Y + idx + M, mask=mm, other=0.0).to(tl.float32)
        inc = ((a + b) * SCALE).to(Y.dtype.element_ty)
        s = inc if LOW else inc.to(tl.float32)
        for j in tl.static_range(0, LOG):
            g = tl.gather(s, tl.maximum(kbi - (1 << j), 0), 0)
            if LOW:
                s = (s + tl.where(kbi >= (1 << j), g, z16)).to(Y.dtype.element_ty)
            else:
                s = s + tl.where(kbi >= (1 << j), g, z32)
        if TILES > 1:
            if LOW:
                s = (s + carry[None, :]).to(Y.dtype.element_ty)
                carry = c16 + tl.sum(tl.where(kbi == KB - 1, s, z16), 0)
            else:
                s = s + carry[None, :]
                carry = c32 + tl.sum(tl.where(kbi == KB - 1, s, z32), 0)
        tl.store(O + idx, s.to(O.dtype.element_ty), mask=mm)


def bench(fn, iters=20):
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
    # A) launch overhead
    x = torch.empty(64, dtype=torch.float32, device='npu')
    rep['trivial_ms'] = bench(lambda: _trivial[(64,)](x), 200)
    print('A trivial launch ms', rep['trivial_ms'], flush=True)

    # B) last-axis doubling alternatives
    for name, shape, dt in (('t0', [128, 1024], torch.float16),
                            ('t1', [1024, 4096], torch.float16)):
        g = torch.Generator().manual_seed(1729)
        y = torch.randn(shape, generator=g).to(dtype=dt, device='npu')
        n1 = shape[1] - 1
        out = torch.empty(shape[0], n1, dtype=dt, device='npu')
        ref = torch.cumulative_trapezoid(y, dx=1, dim=-1)
        blk = triton.next_power_of_2(n1)
        log = blk.bit_length() - 1
        for nw in (1, 2, 4):
            try:
                ms = bench(lambda: cand._scan_vec[(shape[0],)](y, out, 0.5, shape[1],
                                                               blk, log, 1, 1,
                                                               num_warps=nw), 20)
                ms = min(ms, bench(lambda: cand._scan_vec[(shape[0],)](y, out, 0.5,
                                shape[1], blk, log, 1, 1, num_warps=nw), 20))
                rep['%s_gather_nw%d_ms' % (name, nw)] = ms
                print('B', name, 'gather nw', nw, ms, flush=True)
            except Exception as exc:  # noqa: BLE001
                print('B', name, 'nw', nw, 'fail', str(exc)[:200], flush=True)
        scr = torch.empty(shape[0] * blk, dtype=dt, device='npu')
        try:
            cand._scan_vec[(shape[0],)](y, out, 0.5, shape[1], blk, log, 1, 1)
            _hs_scratch[(shape[0],)](y, out, scr, 0.5, shape[1], blk, log, 1)
            torch.npu.synchronize()
            rep['%s_scratch_num' % name] = stat(ref, out, 0.015, 0.015)
            rep['%s_scratch_ms' % name] = bench(
                lambda: _hs_scratch[(shape[0],)](y, out, scr, 0.5, shape[1], blk, log, 1), 20)
            print('B', name, 'scratch', rep['%s_scratch_num' % name],
                  rep['%s_scratch_ms' % name], flush=True)
        except Exception as exc:  # noqa: BLE001
            print('B', name, 'scratch fail', str(exc)[:300], flush=True)
        del y, out, ref

    # C) dim0 path alternatives on [1024,128]
    for dtname, dt in (('f16', torch.float16), ('f32', torch.float32)):
        g = torch.Generator().manual_seed(1729)
        y = torch.randn([1024, 128], generator=g).to(dtype=dt, device='npu')
        ref = torch.cumulative_trapezoid(y, dx=1, dim=0)
        rtol, atol = (1e-4, 1e-4) if dt == torch.float32 else (0.015, 0.015)
        out = torch.empty(1023, 128, dtype=dt, device='npu')
        # C1 current
        ms = bench(lambda: cand.run(y, dx=1, dim=0), 5)
        cand.run(y, dx=1, dim=0)
        torch.npu.synchronize()
        rep['%s_cur' % dtname] = {'ms': ms, 'num': stat(ref, cand.run(y, dx=1, dim=0), rtol, atol)}
        print('C', dtname, 'cur', rep['%s_cur' % dtname], flush=True)
        # C2 transpose + vec + transpose
        yt = torch.empty(128, 1024, dtype=dt, device='npu')
        ot = torch.empty(128, 1023, dtype=dt, device='npu')
        try:
            def run_tr():
                _transpose2d[(16, 2)](y, yt, 1024, 128, 128, 1024, 64, 64)
                cand._scan_vec[(128,)](yt, ot, 0.5, 1024, 1024, 10, 1 if dt != torch.float32 else 0, 1)
                _transpose2d[(16, 2)](ot, out, 128, 1023, 1023, 128, 64, 64)
            run_tr()
            torch.npu.synchronize()
            rep['%s_tr' % dtname] = {'ms': bench(run_tr, 20), 'num': stat(ref, out, rtol, atol)}
            print('C', dtname, 'transpose', rep['%s_tr' % dtname], flush=True)
        except Exception as exc:  # noqa: BLE001
            print('C', dtname, 'transpose fail', str(exc)[:300], flush=True)
        # C3 chunked carry tile
        for kb, cbw in ((128, 8), (256, 4)):
            try:
                out2 = torch.empty(1023, 128, dtype=dt, device='npu')
                lg = kb.bit_length() - 1
                low = 1 if dt == torch.float16 else 0
                _dim0_tile[(128 // cbw,)](y, out2, 0.5, 1024, 128, kb, lg, cbw, low,
                                          1023 // kb + 1)
                torch.npu.synchronize()
                ms = bench(lambda: _dim0_tile[(128 // cbw,)](y, out2, 0.5, 1024, 128, kb,
                          lg, cbw, low, 1023 // kb + 1), 10)
                rep['%s_tile%d' % (dtname, kb)] = {'ms': ms,
                                                   'num': stat(ref, out2, rtol, atol)}
                print('C', dtname, 'tile', kb, rep['%s_tile%d' % (dtname, kb)], flush=True)
            except Exception as exc:  # noqa: BLE001
                print('C', dtname, 'tile', kb, 'fail', str(exc)[:200], flush=True)
        del y, out, ref

    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, 'perf6.json'), 'w') as fh:
        json.dump(rep, fh, indent=1)


main()
