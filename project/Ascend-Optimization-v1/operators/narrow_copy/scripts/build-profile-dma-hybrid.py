#!/usr/bin/env python3
"""将独立复验通过的 KG profiler 循环复制与小输入 DMA 路径合并。"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    operator = Path(__file__).resolve().parents[1]
    request = operator / 'reports/kg-profile-grid-device-master-20261010/independent/kg-profile-1.request.json'
    result = request.with_name('kg-profile-1.result.json')
    data = json.loads(result.read_text(encoding='utf-8'))
    assert data['status'] == 'PASSED' and data['num_passed'] == data['num_workloads'] == 33
    source = json.loads(request.read_text(encoding='utf-8'))['implementation']['sources'][0]['content']
    tool = operator.parent.parent / 'tools/evaluation/compare-operator-versions.py'
    spec = importlib.util.spec_from_file_location('comparison', tool)
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    assert comparison.load_result(result)['source_sha256'] == hashlib.sha256(source.encode('utf-8')).hexdigest()
    dma = operator / 'candidates/small-dma-hybrid-20261010.py'
    suffix = dma.read_text(encoding='utf-8').split('import ctypes as _ctypes', 1)[1]
    content = source + '\n\n# 小输入复用已验证的 CANN DMA，大输入保留 KG profiler 循环复制。\nimport ctypes as _ctypes' + suffix
    ast.parse(content)
    payload = content.encode('utf-8')
    target = operator / 'candidates/profile-small-dma-hybrid-20261010.py'
    assert not target.exists() or target.read_bytes() == payload, '已有候选字节不同，拒绝覆盖'
    target.write_bytes(payload)
    provenance = {
        'upstream_commit': 'd6a8eec473517a3d68157b208eb9c057eb1d4c50',
        'kg_request_sha256': hashlib.sha256(request.read_bytes()).hexdigest(),
        'kg_source_sha256': hashlib.sha256(source.encode('utf-8')).hexdigest(),
        'dma_parent_sha256': hashlib.sha256(dma.read_bytes()).hexdigest(),
        'candidate_sha256': hashlib.sha256(payload).hexdigest(),
        'change': '小连续片段走 CANN DMA；其余使用 KG profiler 的少量 program 循环复制；归属分析优化组',
    }
    target.with_suffix('.provenance.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(target.name, provenance['candidate_sha256'], flush=True)


if __name__ == '__main__':
    main()
