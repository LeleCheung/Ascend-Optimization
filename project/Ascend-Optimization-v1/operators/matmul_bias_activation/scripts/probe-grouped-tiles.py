#!/usr/bin/env python3
"""在 UB 可容纳的分块中测试 L2 分组复用和小 M tile；不改变精度。"""
import json
import importlib.util
import os
from pathlib import Path

import torch
import triton
from triton.backends.ascend.testing import do_bench_npu


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    baseline = load('baseline', 'profiling-kwide-v5-20261010.py')
    tiles = load('tiles', 'probe-pipeline-tiles.py')
    artifact = Path(os.environ['KGS_DEBUG_ARTIFACTS']) / 'grouped-tiles.json'
    # group 改变独立输出 tile 的执行顺序，不分拆归约或改变累加精度。
    configs = [(128, 128, 256, True, False, 0),
               (128, 128, 256, True, False, 4),
               (128, 128, 256, True, False, 8),
               (128, 128, 256, True, False, 16),
               (64, 128, 256, True, False, 8),
               (64, 128, 512, True, True, 8),
               (128, 64, 256, True, False, 8),
               (128, 64, 512, True, True, 8)]
    rows = []
    torch.manual_seed(20261010)
    for dtype in (torch.float16, torch.float32, torch.bfloat16):
        for n in (2048, 4096):
            a = torch.randn((n, n), device='npu', dtype=dtype)
            b = torch.randn_like(a)
            bias = torch.randn((n,), device='npu', dtype=dtype)
            expected = baseline.run(a, b, bias)
            out = torch.empty_like(expected)
            for bm, bn, bk, multi, saving, group in configs:
                row = dict(dtype=str(dtype), n=n, config=[bm, bn, bk],
                           multibuffer=multi, enable_ubuf_saving=saving, group=group)
                try:
                    def launch():
                        tiles.tuned_pipeline[(triton.cdiv(n, bm) * triton.cdiv(n, bn),)](
                            a, b, bias, out, n, n, n, bm, bn, bk, group,
                            multibuffer=multi, enable_ubuf_saving=saving,
                            unit_flag=True, sync_solver=False)
                    launch()
                    torch.npu.synchronize()
                    row['max_abs_vs_v4'] = (out.float() - expected.float()).abs().max().item()
                    row['comparison_candidate'] = 'profiling-kwide-v5-20261010.py'
                    if row['max_abs_vs_v4'] != 0:
                        raise ValueError('与完整通过的 v5 存在数值差异')
                    measured = do_bench_npu(launch, warmup=3, active=8)
                    row['device_kernel_ms'] = float(measured[0] if isinstance(measured, list) else measured)
                except Exception as error:
                    row['error'] = str(error)[-2500:]
                rows.append(row)
                artifact.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
                print(json.dumps(row, ensure_ascii=False), flush=True)
            del a, b, bias, expected, out


if __name__ == '__main__':
    main()
