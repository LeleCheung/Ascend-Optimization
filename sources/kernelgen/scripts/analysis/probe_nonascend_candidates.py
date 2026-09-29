#!/usr/bin/env python3
"""Host-only counterexamples for the frozen September 2026 candidate audit.

This is an evidence script, not a production admission rule or a GPU emulator.
It executes only reviewed, extracted Python functions; address/AA probes
calculate source expressions on the host and do not claim device reproduction.
"""
from __future__ import annotations
import argparse
import ast
import ctypes
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace


def extracted(source, name):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    return compile(ast.Module(body=[node], type_ignores=[]), name, 'exec')


def probe(cohort):
    rows = {r['id']: r for r in cohort['rows']}
    results = []
    def record(key, kind, observed, expected, detail):
        row = rows[key]
        data = Path(row['candidate_path']).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != row['candidate_sha256']:
            raise ValueError(f'Candidate changed: {key}')
        results.append(dict(id=key, operator=row['operator'], sha256=digest,
                            kind=kind, observed=observed, expected=expected, detail=detail))
    def source(key):
        return Path(rows[key]['candidate_path']).read_text()

    # Run the real early-return branch, without importing Torch/Triton.
    scope = {'torch': SimpleNamespace(float32='float32')}
    exec(extracted(source('tianshu:48'), 'run'), scope)
    value = SimpleNamespace(dtype='float32', value=2)
    out = scope['run'](value, None, None, alpha=2, beta=3)
    assert out is value
    record('tianshu:48', 'extracted_dispatcher', out.value, 3*2+2*3*4,
           'fp32, self=2, batch1=3, batch2=4, alpha=2, beta=3; returns self unchanged')

    scope = {'libdevice': SimpleNamespace(log=math.log)}
    exec(extracted(source('tianshu:34'), '_y0_fp32'), scope)
    libm = ctypes.CDLL('libm.so.6')
    libm.y0.argtypes = [ctypes.c_double]
    libm.y0.restype = ctypes.c_double
    record('tianshu:34', 'extracted_scalar_formula_host_float64',
           scope['_y0_fp32'](10.), libm.y0(10.), 'x=10; independent libm y0, not target fp32 arithmetic')

    t0, t1 = Fraction(1), Fraction(-1, 2)
    for _ in range(2, 14):
        t0, t1 = t1, -t1-t0
    assert '_UNROLL = 12' in source('tianshu:36')
    record('tianshu:36', 'exact_mathematical_counterexample', 1, str(t1),
           'x=1/4, n=13; source selects n<=12 only, leaving res=1')

    # An address >= allocation length is outside the tensor; whether an actual
    # allocator faults is deliberately not inferred from host arithmetic.
    assert 'a = tl.load(a_ptrs)' in source('tianshu:40')
    record('tianshu:40', 'source_address_counterexample', {'a_offset': 15*1024, 'load_enabled': True},
           {'valid_offsets': [0, 15*1024-1]}, 'M=15,N=160,K=1024,float32; BM=32,BK=32, NEED_K=False bypasses M mask')
    for key in ['muxi:45', 'muxi:51']:
        assert 'mask=k_mask[None, :]' in source(key)
        record(key, 'source_address_counterexample', {'a_offset': 15*32, 'load_enabled': True},
               {'valid_offsets': [0, 15*32-1]}, 'M=15,N=160,K=32,float32; BM=64, row=15,k=0 has only K mask')
    m, n, k, block = 495, 5333, 71, 2048
    na, nb = m*k, k*n
    end = ((na+nb+block-1)//block)*block
    assert 'mask_b = offs_b >= 0' in source('moer:32')
    record('moer:32', 'source_address_counterexample', {'last_b_offset': end-1-na, 'excess_elements': end-na-nb},
           {'valid_offsets': [0, nb-1]}, 'Reported M495,N5333,K71 enters fp16 conversion; final B load/store lacks offs_b<nb')
    assert 'if EVEN:' in source('pingtouge:26')
    record('pingtouge:26', 'source_address_counterexample', {'first_invalid_x_offset': 512*1024, 'load_enabled': True},
           {'valid_offsets': [0, 512*1024-1]}, 'M512,N1024,K1024,fp32 => fp16 conversion; both lengths /16384 integral but unequal; grid uses max length, EVEN disables both masks')

    # Execute the actual cache hit path. This isolates cache identity behavior;
    # no fake GPU calculation is presented as numeric validation.
    scope = {'_REPACK_CACHE': {}}
    exec(extracted(source('moer:16'), '_get_repacked'), scope)
    weight = SimpleNamespace(value=3)
    packed = SimpleNamespace(value=3)
    scope['_REPACK_CACHE'][id(weight)] = (weight, packed)
    weight.value = 7
    hit = scope['_get_repacked'](weight, 1, 1, 1)
    assert hit is packed
    record('moer:16', 'extracted_cache_hit', hit.value, weight.value,
           'Same weight object mutated 3->7; cached repack is returned without content/version check or repacking')

    # Independent definition of separable bilinear antialias derivative, away
    # from boundaries: for 16->8, center 7 and scale/support 2.
    weights = {i: max(Fraction(0), 1-abs((Fraction(2*i+1,2)-7)/2)) for i in range(16)}
    total = sum(weights.values())
    gold = {(i,j): a*b/(total*total) for i,a in weights.items() for j,b in weights.items() if a*b}
    assert 'h1 % RH == 0' in source('moer:53')
    record('moer:53', 'independent_aa_derivative', {'nonzero_pixels': 1, 'gradient_at_6_6': 1},
           {'nonzero_pixels': len(gold), 'gradient_at_6_6': str(gold[6,6])},
           '16x16->8x8, align_corners=False, grad_output impulse at (3,3); archived downsample kernel scatters at (6,6). This does not establish which MUSA reference the intern executed.')
    captured = {}
    class Launch:
        def __getitem__(self, grid):
            def capture(*args, **kwargs):
                captured['b_strides'] = args[4]
            return capture
    def tensor(shape, strides):
        return SimpleNamespace(shape=shape, device='host-stub', numel=lambda: math.prod(shape),
                               dim=lambda: len(shape), is_contiguous=lambda: True,
                               stride=lambda d=None: strides if d is None else strides[d])
    scope = {'torch': SimpleNamespace(int64='int64', tensor=lambda v, **kw: list(v)),
             '_leq_strided_kernel': Launch()}
    exec(extracted(source('muxi:17'), 'run'), scope)
    scope['run'](tensor((2,3), (3,1)), tensor((3,), (1,)))
    record('muxi:17', 'extracted_broadcast_dispatcher', captured['b_strides'], [0,1],
           'A(2,3),B(3,): source builds [0,0] instead of right-aligned [0,1]; launch is captured without device execution')
    return {'scope': 'Host source/math evidence only; no device compilation, no exact intern replay.', 'probes': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = probe(json.loads(args.cohort.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(f"Recorded {len(result['probes'])} host counterexamples")

if __name__ == '__main__':
    main()
