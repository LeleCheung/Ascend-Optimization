#!/usr/bin/env python3
"""生成连续输入快路径候选，保留 master 的非连续输入路径。"""
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parent.parent
source = root / "candidates/flaggems-master-20261010.py"
destination = root / "candidates/contiguous-copy-20261010.py"
extra = '''

_master_narrow_copy = narrow_copy


@triton.jit
def narrow_copy_contiguous_kernel(X, Y, TOTAL: tl.constexpr,
                                  DIM: tl.constexpr, POST: tl.constexpr,
                                  START: tl.constexpr, LENGTH: tl.constexpr,
                                  SLAB: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if SLAB:
        source_index = index + START * POST
    else:
        # 编译期常量除数，使布局映射不再执行通用整数除法。
        outer = index // (LENGTH * POST)
        inner = index % (LENGTH * POST)
        source_index = outer * DIM * POST + START * POST + inner
    values = tl.load(X + source_index, index < TOTAL)
    tl.store(Y + index, values, index < TOTAL)


def run(inp, dim, start, length):
    if not inp.is_contiguous():
        return _master_narrow_copy(inp, dim, start, length)
    assert -inp.ndim <= dim < inp.ndim
    dim %= inp.ndim
    size = inp.size(dim)
    if start < 0:
        start %= size
    length = min(length, size - start)
    shape = list(inp.shape)
    shape[dim] = length
    out = torch.empty(shape, dtype=inp.dtype, device=inp.device)
    total = out.numel()
    if total == 0:
        return out
    post = 1
    for axis in range(dim + 1, inp.ndim):
        post *= inp.size(axis)
    outer = inp.numel() // (size * post)
    block = 1024 if total < 65536 else 4096 if total < 1048576 else 16384
    with torch.npu.device(inp.device):
        narrow_copy_contiguous_kernel[(triton.cdiv(total, block),)](
            inp, out, total, size, post, start, length, outer == 1, block)
    return out


narrow_copy = run
'''
candidate = source.read_text(encoding="utf-8") + extra
compile(candidate, str(destination), "exec")
if destination.exists() and destination.read_text(encoding="utf-8") != candidate:
    raise SystemExit("已有不同内容的候选，拒绝覆盖")
destination.write_text(candidate, encoding="utf-8", newline="\n")
print(destination.name, hashlib.sha256(candidate.encode()).hexdigest())
