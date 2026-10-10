#!/usr/bin/env python3
"""从真机宽 tile 诊断选择无数值差异的配置，完整评测另行执行。"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('diagnostic', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    source = root / 'candidates/profiling-k256-dotacc-v4-20261010.py'
    rows = json.loads(args.diagnostic.read_text(encoding='utf-8'))
    selected = {}
    for row in rows:
        latency = row.get('device_kernel_ms')
        if (row.get('error') or row.get('max_abs_vs_v4') != 0
                or not isinstance(latency, (int, float))
                or not math.isfinite(latency) or latency <= 0):
            continue
        key = row['dtype'], row['n']
        if key not in selected or latency < selected[key]['device_kernel_ms']:
            selected[key] = row
    assert len(selected) == 6, '需要三个 dtype × 两个矩阵尺寸的有效真机数据'
    probe = root / 'scripts/probe-pipeline-tiles.py'
    kernel = '@triton.jit' + probe.read_text(encoding='utf-8').split('@triton.jit', 1)[1].split('\ndef main():', 1)[0]
    kernel = kernel.replace('def tuned_pipeline(', 'def mba_wide_pipeline_kernel(')
    table = {key: (*row['config'], row['multibuffer'], row['enable_ubuf_saving'], row['group'])
             for key, row in sorted(selected.items())}
    text = source.read_text(encoding='utf-8') + '\n\n' + kernel
    text += '\n_v4_mba = run\n_WIDE_TILES = ' + repr(table) + '\n'
    text += '''

def run(input, weight, bias):
    m, k = input.shape
    n = weight.shape[1]
    config = _WIDE_TILES.get((str(input.dtype), m))
    if (config is None or not (m == n == k) or not input.is_contiguous()
            or not weight.is_contiguous() or not bias.is_contiguous()):
        return _v4_mba(input, weight, bias)
    bm, bn, bk, multi, saving, group = config
    bias = bias.reshape(-1)
    out = torch.empty((m, n), dtype=input.dtype, device=input.device)
    with torch_device_fn.device(input.device):
        mba_wide_pipeline_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
            input, weight, bias, out, m, n, k, bm, bn, bk, group,
            multibuffer=multi, enable_ubuf_saving=saving,
            unit_flag=True, sync_solver=False)
    return out


matmul_bias_activation = run
'''
    target = root / 'candidates/profiling-kwide-v5-20261010.py'
    compile(text, str(target), 'exec')
    assert not target.exists() or target.read_bytes() == text.encode('utf-8')
    target.write_bytes(text.encode('utf-8'))
    provenance = {
        'upstream_commit': 'd6a8eec473517a3d68157b208eb9c057eb1d4c50',
        'parent_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'diagnostic_sha256': hashlib.sha256(args.diagnostic.read_bytes()).hexdigest(),
        'candidate_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'selected': list(selected.values()),
        'verification': '诊断与 v4 最大绝对差为零，待完整 42 项评测。',
    }
    target.with_suffix('.provenance.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(target.name, provenance['candidate_sha256'])


if __name__ == '__main__':
    main()
