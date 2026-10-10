#!/usr/bin/env python3
"""筛选 UB 节省与大 tile；设备计时和数值差异均保留。"""
import importlib.util
import json
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
    baseline = load('baseline', 'profiling-k256-dotacc-v4-20261010.py')
    tiles = load('tiles', 'probe-pipeline-tiles.py')
    artifact = Path(os.environ['KGS_DEBUG_ARTIFACTS']) / 'wide-tiles.json'
    rows = []
    torch.manual_seed(20261010)
    # BM, BN, BK, 多缓冲, UB 节省, 分组。不降低精度，不修改编译器。
    configs = [(128, 128, 256, True, False, 0),
               (128, 128, 512, True, True, 0),
               (128, 256, 128, True, True, 0),
               (256, 128, 128, True, True, 0),
               (128, 256, 256, False, True, 0),
               (256, 128, 256, False, True, 0),
               (128, 256, 128, False, True, 8),
               (256, 128, 128, False, True, 8)]
    for dtype in (torch.float16, torch.float32, torch.bfloat16):
        for n in (1024, 4096):
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
