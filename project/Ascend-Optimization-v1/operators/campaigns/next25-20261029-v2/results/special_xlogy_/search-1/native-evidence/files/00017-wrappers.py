"""Pinpoint the Python-wrapper cost inside run() for special_xlogy_."""
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
    with open(os.path.join(ART, 'wrappers.json'), 'w') as fh:
        json.dump(res, fh, indent=1, sort_keys=True)


def timed(fn, reps=1500, warmup=200):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    t2 = time.perf_counter()
    return {'submit_us': round((t1 - t0) / reps * 1e6, 3),
            'total_us': round((t2 - t0) / reps * 1e6, 3)}


def mk(n, dt):
    a = (torch.randn(n, device=DEV, dtype=dt).abs() + 0.05)
    b = (torch.randn(n, device=DEV, dtype=dt).abs() + 0.05)
    return a, b


N = 1024
DT = torch.float16
A, B = mk(N, DT)
cand.run(A, B)          # populate the compiled-entry cache
torch.npu.synchronize()
key = (A.dtype, B.dtype, A.device.index, N)
res['entry_keys'] = [str(k) for k in cand._ENTRIES]
launch, fn_, meta = cand._ENTRIES[key]
BLK = cand._BLOCK
GRID = triton.cdiv(N, BLK)
stream_cached = cand._stream(A.device.index)
_ENTRIES = cand._ENTRIES
_stream = cand._stream
_cdiv = triton.cdiv
_dev_index = A.device.index

w = {}
w['v0_cand_run'] = timed(lambda: cand.run(A, B))


def v_body_only(A, B):
    n = A.numel()
    if not n:
        return A
    if A.is_contiguous() and B.is_contiguous() and A.shape == B.shape:
        pa, pb = A.data_ptr(), B.data_ptr()
        if not ((pa | pb) & 15):
            e = _ENTRIES.get((A.dtype, B.dtype, A.device.index, n))
            if e is not None:
                return A
    return A


w['v1_wrapper_no_launch'] = timed(lambda: v_body_only(A, B))


def v_cached_stream(A, B):
    pa, pb = A.data_ptr(), B.data_ptr()
    launch(GRID, 1, 1, stream_cached, fn_, meta, None, None, None, pa, pb, N, BLK)
    return A


w['v2_launch_cached_stream'] = timed(lambda: v_cached_stream(A, B))


def v_fresh_stream(A, B):
    pa, pb = A.data_ptr(), B.data_ptr()
    launch(GRID, 1, 1, _stream(_dev_index), fn_, meta, None, None, None, pa, pb, N, BLK)
    return A


w['v3_launch_fresh_stream'] = timed(lambda: v_fresh_stream(A, B))

pa0, pb0 = A.data_ptr(), B.data_ptr()
w['v4_raw_launch_literals'] = timed(
    lambda: launch(GRID, 1, 1, stream_cached, fn_, meta, None, None, None, pa0, pb0, N, BLK))
w['v5_stream_and_cdiv_only'] = timed(lambda: (_stream(_dev_index), _cdiv(N, BLK)))
w['v6_dict_get_only'] = timed(lambda: _ENTRIES.get((A.dtype, B.dtype, _dev_index, N)))
w['v7_devindex_only'] = timed(lambda: (A.dtype, A.device.index, A.numel()))
res['wrapper'] = w
dump()

w2 = {}
A2, B2 = mk(N, DT)
w2['jit_subscript_launch'] = timed(lambda: cand._xlogy_flat[(GRID,)](A2, B2, N, BLK))
w2['torch_xlogy_'] = timed(lambda: A2.xlogy_(B2))
res['alt_launch'] = w2
dump()
print(json.dumps(res, indent=1, sort_keys=True))
