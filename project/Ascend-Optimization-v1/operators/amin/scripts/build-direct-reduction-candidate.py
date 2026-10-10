#!/usr/bin/env python3
"""保留固定 master 的通用路径，增加连续单轴直接归约，避免中间轴转置。"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
source = ROOT / "candidates/flaggems-master-dim-adapter-v2-20261010.py"
destination = ROOT / "candidates/direct-axis-reduction-20261010.py"
extra = '''

# 连续单轴直接索引：outer × reduction × inner，读取相邻 inner 元素。
# 非连续或多轴输入仍使用上游通用实现，未替换为 torch.amin。
@triton.jit
def amin_direct_middle_kernel(X, Y, R: tl.constexpr, I: tl.constexpr,
                              BC: tl.constexpr, BR: tl.constexpr):
    outer = tl.program_id(0)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    reds = tl.arange(0, BR)
    acc = tl.full((BR, BC), float('inf'), tl.float32)
    for start in range(tl.cdiv(R, BR)):
        r = start * BR + reds
        values = tl.load(X + outer * R * I + r[:, None] * I + cols[None, :],
                         (r[:, None] < R) & (cols[None, :] < I), float('inf'))
        acc = tl.minimum(acc, values.to(tl.float32))
    value = tl.min(acc, axis=0)
    tl.store(Y + outer * I + cols, value, cols < I)


@triton.jit
def amin_direct_last_kernel(X, Y, O: tl.constexpr, R: tl.constexpr,
                            BO: tl.constexpr, BR: tl.constexpr):
    rows = tl.program_id(0) * BO + tl.arange(0, BO)
    reds = tl.arange(0, BR)
    acc = tl.full((BO, BR), float('inf'), tl.float32)
    for start in range(tl.cdiv(R, BR)):
        r = start * BR + reds
        values = tl.load(X + rows[:, None] * R + r[None, :],
                         (rows[:, None] < O) & (r[None, :] < R), float('inf'))
        acc = tl.minimum(acc, values.to(tl.float32))
    value = tl.min(acc, axis=1)
    tl.store(Y + rows, value, rows < O)


def run(inp, dim=None, keepdim=False):
    dims = [dim] if isinstance(dim, int) else dim
    if dims is None or len(dims) != 1 or not inp.is_contiguous():
        return _master_amin(inp, dim=dims, keepdim=keepdim)
    axis = dims[0] % inp.ndim
    shape = list(inp.shape)
    reduction = shape[axis]
    outer = math.prod(shape[:axis])
    inner = math.prod(shape[axis + 1:])
    out_shape = shape[:]
    if keepdim:
        out_shape[axis] = 1
    else:
        out_shape.pop(axis)
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        if inner == 1:
            br = triton.next_power_of_2(min(reduction, 2048))
            bo = min(4, max(1, 2048 // br))
            amin_direct_last_kernel[(triton.cdiv(outer, bo),)](
                inp, out, outer, reduction, bo, br)
        else:
            bc = min(64, triton.next_power_of_2(inner))
            br = min(128, triton.next_power_of_2(reduction))
            amin_direct_middle_kernel[(outer, triton.cdiv(inner, bc))](
                inp, out, reduction, inner, bc, br)
    return out


amin = run
'''
candidate = source.read_text(encoding="utf-8") + extra
compile(candidate, str(destination), "exec")
if destination.exists() and destination.read_text(encoding="utf-8") != candidate:
    raise SystemExit("候选已存在且内容不同，拒绝覆盖")
destination.write_text(candidate, encoding="utf-8", newline="\n")
print(destination.name, hashlib.sha256(candidate.encode()).hexdigest())
