#!/usr/bin/env python3
"""根据复制任务诊断，为小连续片段使用 CANN DMA，大输入保留 Triton。"""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
parent = root / 'candidates/contiguous-copy-20261010.py'
text = parent.read_text(encoding='utf-8')
text += '''

import ctypes as _ctypes

try:
    _acl_rt = _ctypes.CDLL("libacl_rt.so")
    _acl_rt.aclrtMemcpyAsync.argtypes = [
        _ctypes.c_void_p, _ctypes.c_size_t, _ctypes.c_void_p,
        _ctypes.c_size_t, _ctypes.c_int, _ctypes.c_void_p,
    ]
    _acl_rt.aclrtMemcpyAsync.restype = _ctypes.c_int
except OSError:
    _acl_rt = None

_triton_contiguous_copy = run


def run(inp, dim, start, length):
    # 只替换单段连续复制；一般维度和布局沿用父候选。
    if not (-inp.ndim <= dim < inp.ndim):
        return _triton_contiguous_copy(inp, dim, start, length)
    axis = dim % inp.ndim
    if _acl_rt is None or axis != 0 or not inp.is_contiguous():
        return _triton_contiguous_copy(inp, dim, start, length)
    size = inp.shape[0]
    normalized_start = start % size if start < 0 and size else start
    count = min(length, size - normalized_start)
    if count <= 0 or normalized_start < 0:
        return _triton_contiguous_copy(inp, dim, start, length)
    post = inp.numel() // size
    elements = count * post
    if elements > 65536:
        return _triton_contiguous_copy(inp, dim, start, length)
    with torch.npu.device(inp.device):
        out = inp.new_empty((count, *inp.shape[1:]))
        amount = elements * inp.element_size()
        source = inp.data_ptr() + normalized_start * post * inp.element_size()
        stream = torch.npu.current_stream(inp.device).npu_stream
        status = _acl_rt.aclrtMemcpyAsync(
            _ctypes.c_void_p(out.data_ptr()), amount,
            _ctypes.c_void_p(source), amount, 3, _ctypes.c_void_p(stream))
        if status != 0:
            raise RuntimeError(f"aclrtMemcpyAsync failed: {status}")
    return out


narrow_copy = run
'''
target = root / 'candidates/small-dma-hybrid-20261010.py'
compile(text, str(target), 'exec')
assert not target.exists() or target.read_bytes() == text.encode()
target.write_bytes(text.encode())
target.with_suffix('.provenance.json').write_text(json.dumps({
    'upstream_commit': 'd6a8eec473517a3d68157b208eb9c057eb1d4c50',
    'parent_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
    'candidate_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
    'implementation_kind': 'CANN DMA + Triton 混合实现',
    'basis': '小输入 PyTorch trace 为 MEMCPY_ASYNC，约 0.6us；大输入为 TensorMove AIV kernel，分别处理',
    'verification': '待完整 33 项评测；不得作为纯 Triton 成果',
}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(target.name)
