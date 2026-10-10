#!/usr/bin/env python3
"""从逐值通过的真机诊断生成 v5；不覆盖已有候选，完整评测另行执行。"""
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
    source = root / 'candidates/direct-axis-tiles-v4-20261010.py'
    probe = root / 'scripts/probe-native-reduction.py'
    rows = json.loads(args.diagnostic.read_text(encoding='utf-8'))
    selected = {}
    for row in rows:
        latency = row.get('device_kernel_ms')
        if (row.get('error') or row.get('equal') is not True
                or not isinstance(latency, (int, float))
                or not math.isfinite(latency) or latency <= 0):
            continue
        key = (row['dtype'], tuple(row['shape']))
        if key not in selected or latency < selected[key]['device_kernel_ms']:
            selected[key] = row
    assert len(selected) == 9, '需要三个 dtype × 三个形状的有效真机数据'
    kernels = probe.read_text(encoding='utf-8').split('@triton.jit', 1)[1].split('\ndef main():', 1)[0]
    kernels = '@triton.jit' + kernels
    kernels = kernels.replace('def middle(', 'def amin_native_middle_kernel(')
    kernels = kernels.replace('def last(', 'def amin_native_last_kernel(')
    table = {key: (row['config'][0], row['config'][1], row['native'])
             for key, row in sorted(selected.items())}
    text = source.read_text(encoding='utf-8') + '\n\n' + kernels
    text += '\n# 分块来自设备 kernel 计时；全归约及未覆盖形状沿用 v4。\n'
    text += '_v4_amin = run\n_NATIVE_TILES = ' + repr(table) + '\n'
    text += '''

def run(inp, dim=None, keepdim=False):
    dims = [dim] if isinstance(dim, int) else dim
    config = _NATIVE_TILES.get((str(inp.dtype), tuple(inp.shape)))
    if (config is None or not inp.is_contiguous() or dims is None
            or len(dims) != 1 or dims[0] % inp.ndim != 1):
        return _v4_amin(inp, dim=dim, keepdim=keepdim)
    block, br, native = config
    shape = list(inp.shape)
    out_shape = shape[:]
    if keepdim:
        out_shape[1] = 1
    else:
        out_shape.pop(1)
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        if inp.ndim == 2:
            amin_native_last_kernel[(triton.cdiv(shape[0], block),)](
                inp, out, shape[0], shape[1], block, br, native,
                multibuffer=False)
        else:
            amin_native_middle_kernel[(shape[0], triton.cdiv(shape[2], block))](
                inp, out, shape[1], shape[2], block, br, native,
                multibuffer=False)
    return out


amin = run
'''
    target = root / 'candidates/direct-axis-native-v5-20261010.py'
    compile(text, str(target), 'exec')
    assert not target.exists() or target.read_bytes() == text.encode('utf-8')
    target.write_bytes(text.encode('utf-8'))
    provenance = {
        'upstream_commit': 'd6a8eec473517a3d68157b208eb9c057eb1d4c50',
        'parent_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'diagnostic_sha256': hashlib.sha256(args.diagnostic.read_bytes()).hexdigest(),
        'candidate_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'selected': list(selected.values()),
        'verification': '诊断逐值一致，待完整 27 项评测；未修改基线或测试。',
    }
    target.with_suffix('.provenance.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(target.name, provenance['candidate_sha256'])


if __name__ == '__main__':
    main()
