#!/usr/bin/env python3
"""对本轮连续复制候选分别测设备计时、host enqueue 和同步调用时间。"""
import hashlib
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path

import torch
import triton
from triton.backends.ascend.testing import do_bench_npu

DEVICE_SCOPE = 'device_kernel'


def device_us(function):
    value = do_bench_npu(function, warmup=5, active=30)
    return 1000 * float(value[0] if isinstance(value, list) else value)


def host_sample(function):
    torch.npu.synchronize()
    start = time.perf_counter_ns()
    value = function()
    enqueued = time.perf_counter_ns()
    torch.npu.synchronize()
    completed = time.perf_counter_ns()
    return value, {'enqueue_us': (enqueued - start) / 1000,
                   'synchronized_call_us': (completed - start) / 1000}


def main():
    source = Path('contiguous-copy-20261010.py')
    spec = importlib.util.spec_from_file_location('current_candidate', source)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    output = Path(os.environ['KGS_DEBUG_ARTIFACTS']) / 'timing-scopes.json'
    result = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'device_scope': DEVICE_SCOPE,
              'scope': '设备侧计时和独立 host 同步采样；两者不相减推导 host 开销', 'cases': []}
    torch.manual_seed(20261010)
    for dtype in (torch.float16, torch.float32, torch.bfloat16):
        for shape in ((64, 64), (1024, 65536)):
            x = torch.randn(shape, dtype=dtype, device='npu')
            start, length = shape[0] // 4, shape[0] // 2
            expected = torch.narrow_copy(x, 0, start, length)
            actual = candidate.run(x, 0, start, length)
            assert torch.equal(expected, actual)
            out = torch.empty_like(actual)
            total, post = out.numel(), shape[1]
            block = 1024 if total < 65536 else 4096 if total < 1048576 else 16384
            def launch():
                candidate.narrow_copy_contiguous_kernel[(triton.cdiv(total, block),)](
                    x, out, total, shape[0], post, start, length, True, block)
                return out
            launch()
            torch.npu.synchronize()
            assert torch.equal(out, expected)
            functions = {'candidate': lambda: candidate.run(x, 0, start, length),
                         'pytorch': lambda: torch.narrow_copy(x, 0, start, length),
                         'allocated_jit_launch': launch,
                         'allocation': lambda: torch.empty_like(out)}
            row = {'dtype': str(dtype), 'shape': shape, 'dim': 0, 'start': start,
                   'length': length, 'equal': True,
                   'logical_bytes': 2 * total * x.element_size(), 'host': {}}
            for name in ('candidate', 'pytorch', 'allocated_jit_launch'):
                row[name + '_' + DEVICE_SCOPE + '_us'] = device_us(functions[name])
            samples = {name: [] for name in functions}
            for iteration in range(25):
                names = list(functions)
                if iteration % 2:
                    names.reverse()
                for name in names:
                    value, sample = host_sample(functions[name])
                    if iteration >= 5:
                        samples[name].append(sample)
                    del value
            for name, observations in samples.items():
                row['host'][name] = {key: {'median_us': statistics.median(s[key] for s in observations),
                                          'min_us': min(s[key] for s in observations),
                                          'max_us': max(s[key] for s in observations)}
                                     for key in ('enqueue_us', 'synchronized_call_us')}
            result['cases'].append(row)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(row, ensure_ascii=False), flush=True)
            del x, expected, actual, out


if __name__ == '__main__':
    main()
