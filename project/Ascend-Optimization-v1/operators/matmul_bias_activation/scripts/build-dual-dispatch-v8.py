#!/usr/bin/env python3
"""复用完整通过的 KG 双 tile 内核，与 v6 的其余路径合并。"""
import ast
import hashlib
import json
from pathlib import Path


def main():
    operator = Path(__file__).resolve().parents[1]
    parent = operator / 'candidates/profiling-kgroup-v6-20261010.py'
    request = operator / 'reports/kg-no-profile-master-20261010/independent/kg-no-profile-1.request.json'
    source = json.loads(request.read_text(encoding='utf-8'))['implementation']['sources'][0]['content']
    tree = ast.parse(source)
    kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'mba_dual_kernel')
    fragment = '@triton.jit\n' + ast.get_source_segment(source, kernel)
    fragment = fragment.replace('def mba_dual_kernel(', 'def mba_dual_dispatch_v8_kernel(', 1)
    wrapper = '''

_v6_dispatch = run


def run(input, weight, bias):
    m, k = input.shape
    n = weight.shape[1]
    # 规则半精度矩阵一次复用 A 面板计算两个相邻 N tile。
    # 其余形状和 dtype 保留已完整通过的 v6。
    if not (input.dtype in (torch.float16, torch.bfloat16)
            and m % 128 == 0 and n % 256 == 0 and k % 256 == 0
            and input.is_contiguous() and weight.is_contiguous()
            and bias.is_contiguous() and bias.numel() == n):
        return _v6_dispatch(input, weight, bias)
    bias = bias.reshape(-1)
    out = torch.empty((m, n), device=input.device, dtype=input.dtype)
    with torch_device_fn.device(input.device):
        mba_dual_dispatch_v8_kernel[((m // 128) * (n // 256),)](
            input, weight, bias, out, m, n, k,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1), bias.stride(0),
            out.stride(0), out.stride(1), 128, 128, 256)
    return out


matmul_bias_activation = run
'''
    result = parent.read_text(encoding='utf-8') + '\n\n' + fragment + wrapper
    ast.parse(result)
    target = operator / 'candidates/profiling-dual-dispatch-v8-20261010.py'
    payload = result.encode('utf-8')
    assert not target.exists() or target.read_bytes() == payload
    target.write_bytes(payload)
    provenance = {
        'upstream_commit': 'd6a8eec473517a3d68157b208eb9c057eb1d4c50',
        'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
        'kg_request_sha256': hashlib.sha256(request.read_bytes()).hexdigest(),
        'kg_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'candidate_sha256': hashlib.sha256(payload).hexdigest(),
        'change': '规则半精度走 KG 无 profiler 双 N tile A 面板复用路径；其余保留 v6；等待完整独立复验',
    }
    target.with_suffix('.provenance.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(target.name, provenance['candidate_sha256'])


if __name__ == '__main__':
    main()
