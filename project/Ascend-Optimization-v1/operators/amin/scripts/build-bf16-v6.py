#!/usr/bin/env python3
"""从 bf16 真机诊断生成新候选，保留 v5 其他路径。"""
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
    parent = root / 'candidates/direct-axis-native-v5-20261010.py'
    probe = root / 'scripts/probe-bf16-cast.py'
    rows = json.loads(args.diagnostic.read_text(encoding='utf-8'))
    selected = {}
    for row in rows:
        latency = row.get('device_kernel_ms')
        if (row.get('error') or row.get('equal') is not True
                or row.get('dtype') != 'torch.bfloat16'
                or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0):
            continue
        key = tuple(row['shape'])
        if key not in selected or latency < selected[key]['device_kernel_ms']:
            selected[key] = row
    assert set(selected) == {(4096, 4096), (64, 512, 512), (1024, 1024, 1024)}
    kernels = '@triton.jit' + probe.read_text(encoding='utf-8').split('@triton.jit', 1)[1].split('\ndef main():', 1)[0]
    kernels = kernels.replace('def middle(', 'def amin_bf16_cast_middle_kernel(')
    kernels = kernels.replace('def last(', 'def amin_bf16_cast_last_kernel(')
    table = {shape: (r['config'][0], r['config'][1], r['native']) for shape, r in sorted(selected.items())}
    text = parent.read_text(encoding='utf-8') + '\n\n' + kernels
    text += '\n_v5_bf16_parent = run\n_BF16_CAST_TILES = ' + repr(table) + '\n'
    text += '''

def run(inp, dim=None, keepdim=False):
    dims = [dim] if isinstance(dim, int) else dim
    config = _BF16_CAST_TILES.get(tuple(inp.shape))
    if (inp.dtype != torch.bfloat16 or config is None or not inp.is_contiguous()
            or dims is None or len(dims) != 1 or dims[0] % inp.ndim != 1):
        return _v5_bf16_parent(inp, dim=dim, keepdim=keepdim)
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
            amin_bf16_cast_last_kernel[(triton.cdiv(shape[0], block),)](
                inp, out, shape[0], shape[1], block, br, native, multibuffer=False)
        else:
            amin_bf16_cast_middle_kernel[(shape[0], triton.cdiv(shape[2], block))](
                inp, out, shape[1], shape[2], block, br, native, multibuffer=False)
    return out


amin = run
'''
    target = root / 'candidates/direct-axis-bf16-v6-20261010.py'
    compile(text, str(target), 'exec')
    assert not target.exists() or target.read_bytes() == text.encode('utf-8')
    target.write_bytes(text.encode('utf-8'))
    provenance = {'upstream_commit': 'd6a8eec473517a3d68157b208eb9c057eb1d4c50',
                  'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
                  'diagnostic_sha256': hashlib.sha256(args.diagnostic.read_bytes()).hexdigest(),
                  'candidate_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                  'selected': list(selected.values()),
                  'verification': '与 PyTorch 和 v5 逐值一致；待完整 27 项评测。'}
    target.with_suffix('.provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(target.name, provenance['candidate_sha256'])


if __name__ == '__main__':
    main()
