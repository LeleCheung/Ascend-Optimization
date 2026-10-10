#!/usr/bin/env python3
"""验证 bf16 minimum 后显式恢复循环类型，筛选分块；最终仍须完整评测。"""
import importlib.util
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.backends.ascend.testing import do_bench_npu


@triton.jit
def middle(X, Y, R: tl.constexpr, I: tl.constexpr,
           BC: tl.constexpr, BR: tl.constexpr, NATIVE: tl.constexpr):
    outer = tl.program_id(0)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    reds = tl.arange(0, BR)
    if NATIVE:
        acc = tl.full((BR, BC), float('inf'), X.dtype.element_ty)
    else:
        acc = tl.full((BR, BC), float('inf'), tl.float32)
    for start in range(tl.cdiv(R, BR)):
        r = start * BR + reds
        values = tl.load(X + outer * R * I + r[:, None] * I + cols[None, :],
                         (r[:, None] < R) & (cols[None, :] < I), float('inf'))
        # minimum 选取已有输入值，转回 bf16 不引入求和的累积舍入。
        acc = tl.minimum(acc, values.to(acc.dtype)).to(acc.dtype)
    value = tl.min(acc, axis=0)
    tl.store(Y + outer * I + cols, value, cols < I)


@triton.jit
def last(X, Y, O: tl.constexpr, R: tl.constexpr,
         BO: tl.constexpr, BR: tl.constexpr, NATIVE: tl.constexpr):
    rows = tl.program_id(0) * BO + tl.arange(0, BO)
    reds = tl.arange(0, BR)
    if NATIVE:
        acc = tl.full((BO, BR), float('inf'), X.dtype.element_ty)
    else:
        acc = tl.full((BO, BR), float('inf'), tl.float32)
    for start in range(tl.cdiv(R, BR)):
        r = start * BR + reds
        values = tl.load(X + rows[:, None] * R + r[None, :],
                         (rows[:, None] < O) & (r[None, :] < R), float('inf'))
        acc = tl.minimum(acc, values.to(acc.dtype)).to(acc.dtype)
    value = tl.min(acc, axis=1)
    tl.store(Y + rows, value, rows < O)


def main():
    spec = importlib.util.spec_from_file_location('v5', 'direct-axis-native-v5-20261010.py')
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    torch.manual_seed(20261010)
    rows = []
    artifact = Path(os.environ['KGS_DEBUG_ARTIFACTS']) / 'bf16-cast.json'
    for shape in ((4096, 4096), (64, 512, 512), (1024, 1024, 1024)):
        x = torch.randn(shape, device='npu', dtype=torch.bfloat16)
        expected = torch.amin(x, dim=1)
        previous = baseline.run(x, dim=1)
        assert torch.equal(previous, expected)
        out = torch.empty_like(expected)
        is_last = len(shape) == 2
        configs = ([(16, 256), (8, 1024), (4, 2048), (2, 4096)] if is_last else
                   [(256, 16), (512, 16), (512, 32), (1024, 8), (1024, 16)])
        measured = do_bench_npu(lambda: baseline.run(x, dim=1), warmup=3, active=8)
        baseline_ms = float(measured[0] if isinstance(measured, list) else measured)
        for block, br in configs:
            for native in (False, True):
                row = dict(dtype='torch.bfloat16', shape=shape, config=[block, br],
                           native=native, multibuffer=False, baseline_device_kernel_ms=baseline_ms)
                try:
                    if is_last:
                        def launch():
                            last[(triton.cdiv(shape[0], block),)](
                                x, out, shape[0], shape[1], block, br, native, multibuffer=False)
                    else:
                        def launch():
                            middle[(shape[0], triton.cdiv(shape[2], block))](
                                x, out, shape[1], shape[2], block, br, native, multibuffer=False)
                    launch()
                    torch.npu.synchronize()
                    row['equal'] = bool(torch.equal(out, expected) and torch.equal(out, previous))
                    if not row['equal']:
                        raise ValueError('与 PyTorch 或 v5 逐值结果不一致')
                    measured = do_bench_npu(launch, warmup=3, active=8)
                    row['device_kernel_ms'] = float(measured[0] if isinstance(measured, list) else measured)
                except Exception as error:
                    row['error'] = str(error)[-2500:]
                rows.append(row)
                artifact.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
                print(json.dumps(row, ensure_ascii=False), flush=True)
        del x, expected, previous, out


if __name__ == '__main__':
    main()
