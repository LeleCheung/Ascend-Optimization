#!/usr/bin/env python3
"""生成连续输入全量归约候选；每阶段最多归约 16384 个元素。"""
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parent.parent
source = root / "candidates/direct-axis-reduction-20261010.py"
destination = root / "candidates/direct-axis-global-v3-20261010.py"
extra = '''

_direct_axis_amin = run

@triton.jit
def amin_global_chunk_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, offsets < N, float('inf'))
    result = tl.min(values.to(tl.float32), axis=0)
    tl.store(Y + tl.program_id(0), result)


def run(inp, dim=None, keepdim=False):
    dims = [dim] if isinstance(dim, int) else dim
    if not inp.is_contiguous() or (dims is not None and len(dims) != 0):
        return _direct_axis_amin(inp, dim=dims, keepdim=keepdim)
    count = inp.numel()
    if count == 0:
        return _master_amin(inp, dim=dims, keepdim=keepdim)
    out = torch.empty([1] * inp.ndim if keepdim else [],
                      dtype=inp.dtype, device=inp.device)
    current = inp
    # 阶段数随输入增长，避免第二阶段生成过大的单 program。
    with torch_device_fn.device(inp.device):
        while True:
            block = min(16384, triton.next_power_of_2(count))
            chunks = triton.cdiv(count, block)
            target = out if chunks == 1 else torch.empty(
                (chunks,), dtype=inp.dtype, device=inp.device)
            amin_global_chunk_kernel[(chunks,)](current, target, count, block)
            if chunks == 1:
                break
            current, count = target, chunks
    return out


amin = run
'''
candidate = source.read_text(encoding="utf-8") + extra
compile(candidate, str(destination), "exec")
if destination.exists() and destination.read_text(encoding="utf-8") != candidate:
    raise SystemExit("候选已存在且不同，拒绝覆盖")
destination.write_text(candidate, encoding="utf-8", newline="\n")
print(destination.name, hashlib.sha256(candidate.encode()).hexdigest())
