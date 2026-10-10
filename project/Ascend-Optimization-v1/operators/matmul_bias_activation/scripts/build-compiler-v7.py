#!/usr/bin/env python3
"""保留父候选回退，仅用逐值一致且真机更快的规则矩阵配置。"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostic', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    parent = root / 'candidates/profiling-kwide-v5-20261010.py'
    probe = root / 'scripts/probe-compiler-pipeline.py'
    rows = json.loads(args.diagnostic.read_text(encoding='utf-8'))
    selected = {}
    covered = set()
    for row in rows:
        key = row['dtype'], row['n']
        latency = row.get('device_kernel_ms')
        if (row.get('error') or row.get('max_abs_vs_parent') != 0
                or type(latency) not in (int, float) or not math.isfinite(latency) or latency <= 0):
            continue
        covered.add(key)
        # 至少快 1%，避免只按一次采样的微小波动替换。
        if latency >= row['baseline_device_kernel_ms'] * 0.99:
            continue
        if key not in selected or latency < selected[key]['device_kernel_ms']:
            selected[key] = row
    expected = {(dtype, n) for dtype in ('torch.float16', 'torch.float32', 'torch.bfloat16')
                for n in (1024, 2048, 4096)}
    assert covered == expected, '需要每个 dtype 与尺寸的有效诊断，失败记录保留'
    kernel = '@triton.jit' + probe.read_text(encoding='utf-8').split('@triton.jit', 1)[1].split('\ndef main():', 1)[0]
    configs = {key: (row['column_major'], row['options']) for key, row in sorted(selected.items())}
    source = parent.read_text(encoding='utf-8') + '\n\n' + kernel
    source += '\n_compiler_v7_parent = run\n_COMPILER_TILES = ' + repr(configs) + '\n'
    source += '''

def run(input, weight, bias):
    m, k = input.shape
    n = weight.shape[1]
    config = _COMPILER_TILES.get((str(input.dtype), m))
    if (config is None or not (m == n == k) or m % 256
            or not input.is_contiguous() or not weight.is_contiguous()
            or not bias.is_contiguous() or bias.numel() != n):
        return _compiler_v7_parent(input, weight, bias)
    column_major, options = config
    out = torch.empty((m, n), dtype=input.dtype, device=input.device)
    with torch_device_fn.device(input.device):
        mba_even_pipeline_kernel[((n // 128) ** 2,)](
            input, weight, bias.reshape(-1), out, n, 128, 128, 256, column_major, **options)
    return out


matmul_bias_activation = run
'''
    target = root / 'candidates/profiling-kcompiler-v7-20261010.py'
    compile(source, str(target), 'exec')
    assert not target.exists() or target.read_bytes() == source.encode('utf-8')
    target.write_bytes(source.encode('utf-8'))
    origin = {'upstream_commit': 'd6a8eec473517a3d68157b208eb9c057eb1d4c50',
              'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
              'diagnostic_sha256': hashlib.sha256(args.diagnostic.read_bytes()).hexdigest(),
              'candidate_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
              'selected': list(selected.values()),
              'verification': '诊断与父候选逐值相同；待完整 42 项评测'}
    target.with_suffix('.provenance.json').write_text(json.dumps(origin, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(target.name, 'selected', len(configs))


if __name__ == '__main__':
    main()
