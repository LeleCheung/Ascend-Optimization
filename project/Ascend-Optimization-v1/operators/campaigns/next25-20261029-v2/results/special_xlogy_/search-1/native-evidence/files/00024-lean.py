"""Locate the residual per-call cost above the raw-launcher floor, plus 1M device check."""
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
res = {'python': sys.version.split()[0]}


def dump():
    with open(os.path.join(ART, 'lean.json'), 'w') as fh:
        json.dump(res, fh, indent=1, sort_keys=True)


def timed(fn, reps=2000, warmup=300):
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

launch, function, metadata, grid, block, device = cand._ENTRIES[(A.dtype, B.dtype, N)]
stream = cand._stream(device)
pa, pb = A.data_ptr(), B.data_ptr()
res['entry'] = {'grid': grid, 'block': block, 'device': device,
                'launch_type': type(launch).__name__}

w = {}
w['L0_cand_run'] = timed(lambda: cand.run(A, B))


def l1_launch_only():
    launch(grid, 1, 1, stream, function, metadata, None, None, None, pa, pb, N, block)


w['L1_launch_literals'] = timed(l1_launch_only)


def l2_full_like_but_launch_only(A, B):
    n = A.numel()
    if not n:
        return A
    if A.is_contiguous() and B.is_contiguous() and A.shape == B.shape:
        entry = cand._ENTRIES.get((A.dtype, B.dtype, n))
        if entry is not None:
            pA = A.data_ptr()
            pB = B.data_ptr()
            if not ((pA | pB) & 15):
                l_, f_, m_, g_, b_, d_ = entry
                l_(g_, 1, 1, stream, f_, m_, None, None, None, pA, pB, n, b_)
                return A
    return A


w['L2_full_checks_cached_stream'] = timed(lambda: l2_full_like_but_launch_only(A, B))


def l3_no_checks(A, B):
    n = N
    entry = cand._ENTRIES.get((A.dtype, B.dtype, n))
    l_, f_, m_, g_, b_, d_ = entry
    pA = A.data_ptr()
    pB = B.data_ptr()
    l_(g_, 1, 1, cand._stream(d_), f_, m_, None, None, None, pA, pB, n, b_)
    return A


w['L3_no_guards'] = timed(lambda: l3_no_checks(A, B))


def l4_key_only(A, B):
    entry = cand._ENTRIES.get((A.dtype, B.dtype, A.numel()))
    return entry


w['L4_key_and_lookup'] = timed(lambda: l4_key_only(A, B))
w['L5_guards_only'] = timed(
    lambda: (A.is_contiguous(), B.is_contiguous(), A.shape == B.shape, A.data_ptr(), B.data_ptr()))
w['L6_stream_only'] = timed(lambda: cand._stream(device))

# tensor-argument launch convention
try:
    comp = cand._xlogy_flat.warmup(A, B, N, block, grid=(1,))
    r = comp.run
    tl_launch = r.launch
    tl_launch(grid, 1, 1, stream, comp.function, comp.packed_metadata,
              None, None, None, A, B, N, block)
    torch.npu.synchronize()
    a2, b2 = mk(N, DT)
    exp = torch.xlogy(a2.clone(), b2)
    tl_launch(grid, 1, 1, stream, comp.function, comp.packed_metadata,
              None, None, None, a2, b2, N, block)
    torch.npu.synchronize()
    ok = bool(torch.allclose(a2.float(), exp.float(), atol=1e-2, rtol=1e-2, equal_nan=True))
    res['tensor_args_ok'] = ok
    if ok:
        w['L7_launch_tensor_args'] = timed(
            lambda: tl_launch(grid, 1, 1, stream, comp.function, comp.packed_metadata,
                              None, None, None, A, B, N, block))
except Exception as exc:  # noqa: BLE001
    res['tensor_args_error'] = repr(exc)[:300]
res['wrapper'] = w
dump()

# 1M bf16 device check: is any of the residual exposed device time?
X, Y = mk(1048576, torch.bfloat16)
dev = {}
for blk in (4096, 8192):
    g = (1048576 + blk - 1) // blk
    dev['flat_%d' % blk] = timed(lambda blk=blk, g=g: cand._xlogy_flat[(g,)](X, Y, 1048576, blk),
                                 reps=600, warmup=100)
dev['torch_xlogy_'] = timed(lambda: X.xlogy_(Y), reps=600, warmup=100)
res['m1_bf16'] = dev
dump()
print(json.dumps(res, indent=1, sort_keys=True))
