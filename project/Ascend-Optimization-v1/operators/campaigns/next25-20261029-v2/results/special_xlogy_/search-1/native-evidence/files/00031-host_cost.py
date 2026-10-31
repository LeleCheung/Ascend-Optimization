"""Bounded host-cost decomposition for special_xlogy_ in-place candidate."""
import json
import os
import sys
import time

import torch
import torch_npu

sys.path.insert(0, 'tmp')
import main as cand  # noqa: E402

ART = os.environ.get('KGS_DEBUG_ARTIFACTS', '.')
DEV = 'npu:0'
res = {}


def timed(fn, reps=400, warmup=80):
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


cases = {
    'f16_1024': (1024, torch.float16),
    'bf16_1M': (1048576, torch.bfloat16),
    'f16_16M': (4096 * 4096, torch.float16),
    'f32_16M': (4096 * 4096, torch.float32),
}

res['per_case'] = {}
for name, (n, dt) in cases.items():
    A, B = mk(n, dt)
    Aref = A.clone()
    r = {}
    r['candidate'] = timed(lambda: cand.run(A, B))
    r['torch_xlogy_'] = timed(lambda: Aref.xlogy_(B))
    res['per_case'][name] = r

# --- component costs on the small fp16 case ---
A, B = mk(1024, torch.float16)
n = 1024
comp = {}
comp['full_run'] = timed(lambda: cand.run(A, B))

launcher = entry = None
key = (A.dtype, B.dtype, A.device.index, n)
if key in cand._ENTRIES:
    entry = cand._ENTRIES[key]
if entry is not None:
    launch, fn_, meta = entry
    stream = cand._stream(A.device.index)
    pa, pb = A.data_ptr(), B.data_ptr()
    comp['raw_launch_4args'] = timed(
        lambda: launch(1, 1, 1, stream, fn_, meta, None, None, None, pa, pb, n, cand._BLOCK))
    comp['raw_launch_2args'] = None
    try:
        launch(1, 1, 1, stream, fn_, meta, None, None, None, pa, pb)
        torch.npu.synchronize()
        ref = (A.clone().float()).clone()
        a2, b2 = mk(1024, torch.float16)
        exp = torch.xlogy(a2.clone(), b2)
        launch(1, 1, 1, cand._stream(0), fn_, meta, None, None, None,
               a2.data_ptr(), b2.data_ptr())
        torch.npu.synchronize()
        ok = bool(torch.allclose(a2.float(), exp.float(), atol=1e-2, rtol=1e-2, equal_nan=True))
        comp['raw_launch_2args'] = {'ok': ok}
        comp['raw_launch_2args']['timing'] = timed(
            lambda: launch(1, 1, 1, stream, fn_, meta, None, None, None, pa, pb))
    except Exception as exc:  # noqa: BLE001
        comp['raw_launch_2args'] = {'error': repr(exc)[:300]}
    comp['stream_query_only'] = timed(lambda: cand._stream(A.device.index))
    comp['metadata_only'] = timed(lambda: (A.dtype, A.device.index, A.numel()))
    comp['data_ptr_only'] = timed(lambda: (A.data_ptr(), B.data_ptr()))
    comp['contig_only'] = timed(lambda: (A.is_contiguous(), B.is_contiguous()))
    comp['launcher_type'] = type(launch).__name__
    comp['launcher_module'] = getattr(type(launch), '__module__', '')
    comp['entry_source_ok'] = True
res['components'] = comp

# --- device-bound sweep of BLOCK on the 4096x4096 cases (python launch path) ---
sweep = {}
for name, dt in (('f16_16M', torch.float16), ('f32_16M', torch.float32)):
    n = 4096 * 4096
    A, B = mk(n, dt)
    row = {}
    for blk in (4096, 8192, 16384):
        grid = (n + blk - 1) // blk
        try:
            row[str(blk)] = timed(lambda blk=blk, grid=grid: cand._xlogy_flat[grid](A, B, n, blk),
                                  reps=120, warmup=30)
        except Exception as exc:  # noqa: BLE001
            row[str(blk)] = {'error': repr(exc)[:200]}
    sweep[name] = row
res['block_sweep_large'] = sweep

with open(os.path.join(ART, 'host_cost.json'), 'w') as fh:
    json.dump(res, fh, indent=1, sort_keys=True)
print(json.dumps(res, indent=1, sort_keys=True))
