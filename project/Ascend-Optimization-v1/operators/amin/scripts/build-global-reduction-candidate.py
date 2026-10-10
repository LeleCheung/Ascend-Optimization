#!/usr/bin/env python3
"""在单轴候选上减少全量归约的 program 数；保留旧候选供比较。"""
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parent.parent
source = root / "candidates/direct-axis-reduction-20261010.py"
destination = root / "candidates/direct-axis-global-v2-20261010.py"
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
    shape = [1] * inp.ndim if keepdim else []
    out = torch.empty(shape, dtype=inp.dtype, device=inp.device)
    block = min(16384, triton.next_power_of_2(count))
    chunks = triton.cdiv(count, block)
    with torch_device_fn.device(inp.device):
        if chunks == 1:
            amin_global_chunk_kernel[(1,)](inp, out, count, block)
        else:
            partial = torch.empty((chunks,), dtype=inp.dtype, device=inp.device)
            amin_global_chunk_kernel[(chunks,)](inp, partial, count, block)
            amin_global_chunk_kernel[(1,)](
                partial, out, chunks, triton.next_power_of_2(chunks))
    return out


amin = run
'''
candidate = source.read_text(encoding="utf-8") + extra
compile(candidate, str(destination), "exec")
if destination.exists() and destination.read_text(encoding="utf-8") != candidate:
    raise SystemExit("候选已存在且不同，拒绝覆盖")
destination.write_text(candidate, encoding="utf-8", newline="\n")
print(destination.name, hashlib.sha256(candidate.encode()).hexdigest())
