"""Diagnostic: compare Triton fused scan vs torch.cumulative_trapezoid on target.

Screening only; eval_round is authoritative.
"""
import importlib.util
import json
import os
import sys
import time

import torch

spec = importlib.util.spec_from_file_location('cand', 'tmp/main.py')
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)

DT = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}

CASES = [
    ('t0', [128, 1024], 'float16', -1, 1, False),
    ('t1', [1024, 4096], 'float16', -1, 1, False),
    ('t2', [1024, 128], 'float16', 0, 1, False),
    ('t3', [128, 1024], 'bfloat16', -1, 1, False),
    ('t4', [1024, 4096], 'bfloat16', -1, 1, False),
    ('t5', [1024, 128], 'bfloat16', 0, 1, False),
    ('t6', [128, 1024], 'float32', -1, 1, False),
    ('t7', [1024, 4096], 'float32', -1, 1, False),
    ('t8', [1024, 128], 'float32', 0, 1, False),
    ('c0', [3, 17], 'float16', -1, 1, False),
    ('c12', [7, 9], 'bfloat16', -1, 1, True),
    ('c18', [7, 9], 'float32', 0, 1, False),
    ('c17', [3, 17], 'float32', -1, 0.25, False),
]


def build(shape, dt, transpose, seed):
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(shape, generator=g).to(dtype=DT[dt], device='npu')
    if transpose:
        y = y.transpose(-1, -2)
    return y


def compare(ref, got, rtol, atol):
    if ref.shape != got.shape:
        return {'shape_mismatch': [list(ref.shape), list(got.shape)]}
    if ref.dtype != got.dtype:
        return {'dtype_mismatch': [str(ref.dtype), str(got.dtype)]}
    a = ref.float()
    b = got.float()
    diff = (a - b).abs()
    tol = atol + rtol * a.abs()
    bad = diff > tol
    nbad = int(bad.sum().item())
    out = {
        'n': int(a.numel()),
        'nbad': nbad,
        'matched_ratio': 1.0 - nbad / max(1, a.numel()),
        'max_abs_diff': float(diff.max().item()) if a.numel() else 0.0,
        'bitexact': bool((a == b).all().item()),
    }
    if nbad:
        flat = bad.flatten().nonzero().flatten()[:5]
        out['worst'] = [
            {'i': int(i),
             'ref': float(a.flatten()[i].item()),
             'got': float(b.flatten()[i].item()),
             'diff': float(diff.flatten()[i].item())}
            for i in flat
        ]
    return out


def bench(fn, iters=30):
    fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def main():
    art = os.environ.get('KGS_DEBUG_ARTIFACTS', '.')
    report = {}
    for (name, shape, dt, dim, dx, tr) in CASES:
        rtol, atol = (0.0001, 0.0001) if dt == 'float32' else (0.015, 0.015)
        y = build(shape, dt, tr, 1729)
        ref = torch.cumulative_trapezoid(y, dx=dx, dim=dim)
        entry = {}
        accs = [1] if dt == 'float32' else [1, 0]
        for acc in accs:
            cand._ACC32 = acc
            try:
                got = cand.run(y, dx=dx, dim=dim)
                entry['acc%d' % acc] = compare(ref, got, rtol, atol)
            except Exception as exc:  # noqa: BLE001
                entry['acc%d' % acc] = {'error': '%s: %s' % (type(exc).__name__, exc)}
        if name.startswith('t'):
            t_ref = bench(lambda: torch.cumulative_trapezoid(y, dx=dx, dim=dim))
            entry['ms_ref'] = t_ref
            cand._ACC32 = 1
            entry['ms_cand'] = bench(lambda: cand.run(y, dx=dx, dim=dim))
        report[name] = entry
        print(name, json.dumps(entry)[:600], flush=True)
    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, 'numerics.json'), 'w') as fh:
        json.dump(report, fh, indent=1)


main()
