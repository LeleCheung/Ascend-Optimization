#!/usr/bin/env python3
"""验证原生精度归约与连续宽列，逐值一致后再筛选设备延迟。"""
import json
import os
import statistics
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
        acc = tl.minimum(acc, values.to(acc.dtype))
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
        acc = tl.minimum(acc, values.to(acc.dtype))
    value = tl.min(acc, axis=1)
    tl.store(Y + rows, value, rows < O)


def main():
    torch.manual_seed(20261010)
    rows = []
    artifact = Path(os.environ['KGS_DEBUG_ARTIFACTS']) / 'native-reduction.json'
    for dtype in (torch.float16, torch.float32, torch.bfloat16):
        for shape in ((4096, 4096), (64, 512, 512), (1024, 1024, 1024)):
            x = torch.randn(shape, device='npu', dtype=dtype)
            expected = torch.amin(x, dim=1)
            out = torch.empty_like(expected)
            last_axis = len(shape) == 2
            configs = ([(16, 256), (8, 1024), (4, 2048), (2, 4096)] if last_axis else
                       [(256, 16), (512, 16), (512, 32), (1024, 8), (1024, 16)])
            for block, br in configs:
                for native in ([False, True] if dtype != torch.float32 else [False]):
                    row = dict(dtype=str(dtype), shape=shape, config=[block, br],
                               native=native, multibuffer=False)
                    try:
                        if last_axis:
                            def launch():
                                last[(triton.cdiv(shape[0], block),)](
                                    x, out, shape[0], shape[1], block, br, native,
                                    multibuffer=False)
                        else:
                            def launch():
                                middle[(shape[0], triton.cdiv(shape[2], block))](
                                    x, out, shape[1], shape[2], block, br, native,
                                    multibuffer=False)
                        launch()
                        torch.npu.synchronize()
                        row['equal'] = bool(torch.equal(out, expected))
                        if not row['equal']:
                            raise ValueError('逐值结果不一致')
                        times = []
                        for _ in range(3):
                            a, b = [torch.npu.Event(enable_timing=True) for _ in range(2)]
                            a.record()
                            for _ in range(5):
                                launch()
                            b.record()
                            torch.npu.synchronize()
                            times.append(a.elapsed_time(b) / 5)
                        row.update(event_ms=times, median_ms=statistics.median(times))
                        # 与 FlagGems core 一致的设备 kernel 口径，避免事件间 CPU 空闲影响筛选。
                        measured = do_bench_npu(launch, warmup=3, active=8)
                        if isinstance(measured, list):
                            measured = measured[0]
                        row['device_kernel_ms'] = float(measured)
                    except Exception as error:
                        row['error'] = str(error)[-2500:]
                    rows.append(row)
                    artifact.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
                    print(json.dumps(row, ensure_ascii=False), flush=True)
            del x, expected, out


if __name__ == '__main__':
    main()
