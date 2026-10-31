"""Compare candidate run() bodies and probe alternative launch entry points."""
import json
import os
import sys
import time

import torch
import torch_npu
import triton

sys.path.insert(0, 'tmp')
import main as cand  # noqa: E402

ART = os.environ.get('KGS_DEBUG_ARTIFACTS', '.')
DEV = 'npu:0'
res = {}


def dump():
    with open(os.path.join(ART, 'frames.json'), 'w') as fh:
        json.dump(res, fh, indent=1, sort_keys=True)


def timed(fn, reps=3000, warmup=400):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    t2 = time.perf_counter()
    return round((t1 - t0) / reps * 1e6, 3), round((t2 - t0) / reps * 1e6, 3)


def mk(n, dt):
    a = (torch.randn(n, device=DEV, dtype=dt).abs() + 0.05)
    b = (torch.randn(n, device=DEV, dtype=dt).abs() + 0.05)
    return a, b


N = 1024
DT = torch.float16
A, B = mk(N, DT)
cand.run(A, B)
torch.npu.synchronize()
entry = cand._ENTRIES[(A.dtype, B.dtype, N)]
launch, function, metadata, grid, block, device = entry
stream = cand._stream(device)
pa, pb = A.data_ptr(), B.data_ptr()
E = cand._ENTRIES

# --- probe alternative launch entry points ---
probe = {'run_dir': [d for d in dir(type(entry[0]))][:0]}
try:
    comp = cand._xlogy_flat.warmup(A, B, N, block, grid=(1,))
    probe['compiled_attrs'] = [a for a in dir(comp) if not a.startswith('_')][:60]
    probe['runner_attrs'] = [a for a in dir(comp.run) if not a.startswith('_')][:60]
    probe['runner_type'] = type(comp.run).__name__
except Exception as exc:  # noqa: BLE001
    probe['error'] = repr(exc)[:200]
res['probe'] = probe


def var_current(A, B):
    n = A.numel()
    if not n:
        return A
    if A.is_contiguous() and B.is_contiguous() and A.shape == B.shape:
        e = E.get((A.dtype, B.dtype, n))
        if e is None:
            e = cand._compile(A, B, n)
            E[(A.dtype, B.dtype, n)] = e
        if e is not None:
            pA = A.data_ptr()
            pB = B.data_ptr()
            if not ((pA | pB) & 15):
                l_, f_, m_, g_, b_, d_ = e
                l_(g_, 1, 1, cand._stream(d_), f_, m_, None, None, None, pA, pB, n, b_)
                return A
        return A
    return cand._generic(A, B, n) or A


def var_key_guard(A, B):
    n = A.numel()
    e = E.get((A.dtype, B.dtype, n,
               A.is_contiguous() and B.is_contiguous() and A.shape == B.shape))
    if e is not None:
        pA = A.data_ptr()
        pB = B.data_ptr()
        if not ((pA | pB) & 15):
            l_, f_, m_, g_, b_, d_ = e
            l_(g_, 1, 1, cand._stream(d_), f_, m_, None, None, None, pA, pB, n, b_)
            return A
    return cand.run(A, B)


def var_no_align(A, B):
    n = A.numel()
    e = E.get((A.dtype, B.dtype, n,
               A.is_contiguous() and B.is_contiguous() and A.shape == B.shape))
    if e is not None:
        pA = A.data_ptr()
        pB = B.data_ptr()
        l_, f_, m_, g_, b_, d_ = e
        l_(g_, 1, 1, stream, f_, m_, None, None, None, pA, pB, n, b_)
        return A
    return A


w = {}
w['V0_cand_run'] = timed(lambda: cand.run(A, B))
w['V1_current_copy'] = timed(lambda: var_current(A, B))
w['V2_key_guard'] = timed(lambda: var_key_guard(A, B))
w['V3_no_align_no_stream'] = timed(lambda: var_no_align(A, B))
res['variants'] = w
dump()

# --- runner-call form vs raw launch ---
alt = {}
try:
    comp = cand._xlogy_flat.warmup(A, B, N, block, grid=(1,))
    r = comp.run
    args = (grid, 1, 1, stream, comp.function, comp.packed_metadata,
            None, None, None, pa, pb, N, block)
    alt['runner_call'] = timed(lambda: r(*args))
    alt['raw_launch'] = timed(lambda: entry[0](*args))
except Exception as exc:  # noqa: BLE001
    alt['error'] = repr(exc)[:200]
res['alt'] = alt
dump()
print(json.dumps(res, indent=1, sort_keys=True))
