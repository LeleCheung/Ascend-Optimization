#!/usr/bin/env python3
"""把单段连续片段的 DMA 路径扩展到所有大小，其余布局保留 Triton。"""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
parent = root / 'candidates/small-dma-hybrid-20261010.py'
text = parent.read_text(encoding='utf-8')
guard = '    if elements > 65536:\n        return _triton_contiguous_copy(inp, dim, start, length)\n'
assert text.count(guard) == 1
text = text.replace(guard, '')
target = root / 'candidates/contiguous-dma-hybrid-20261010.py'
compile(text, str(target), 'exec')
assert not target.exists() or target.read_bytes() == text.encode()
target.write_bytes(text.encode())
target.with_suffix('.provenance.json').write_text(json.dumps({
    'upstream_commit': 'd6a8eec473517a3d68157b208eb9c057eb1d4c50',
    'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
    'candidate_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
    'implementation_kind': 'CANN DMA + Triton 混合实现',
    'basis': '单段连续 narrow 是实际设备内存复制；取消大小门限，测试 DMA 能否优于 AIV TensorMove 与 Triton 复制',
    'verification': '待完整 33 项评测；输出重新分配、实际复制，不返回视图或缓存数据',
}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(target.name)
