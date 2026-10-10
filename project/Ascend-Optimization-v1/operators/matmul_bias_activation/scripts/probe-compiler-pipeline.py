#!/usr/bin/env python3
"""规则矩阵去掉边界 mask，测试当前后端的搬运与多缓冲选项。"""
import importlib.util
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.backends.ascend.testing import do_bench_npu


@triton.jit
def mba_even_pipeline_kernel(A, B, Bias, C, N: tl.constexpr,
                             BM: tl.constexpr, BN: tl.constexpr,
                             BK: tl.constexpr, COLUMN_MAJOR: tl.constexpr):
    pid = tl.program_id(0)
    if COLUMN_MAJOR:
        pm, pn = pid % (N // BM), pid // (N // BM)
    else:
        pm, pn = pid // (N // BN), pid % (N // BN)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = A + rm[:, None] * N + rk[None, :]
    bp = B + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(N // BK):
        a = tl.load(ap)
        b = tl.load(bp)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        ap += BK
        bp += BK * N
    acc += tl.load(Bias + rn)[None, :]
    acc = tl.maximum(acc, acc * 0.0)
    tl.store(C + rm[:, None] * N + rn[None, :], acc)


def main():
    spec = importlib.util.spec_from_file_location('v5', 'profiling-kwide-v5-20261010.py')
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    artifact = Path(os.environ['KGS_DEBUG_ARTIFACTS']) / 'compiler-pipeline.json'
    base_options = dict(multibuffer=True, unit_flag=True, sync_solver=False)
    # 每个选项均在本机 NPUOptions 中存在，使用独立编译缓存；不修改全局环境。
    variants = [
        ('mask-free', {}),
        ('nd2nz-vector', {'enable_nd2nz_on_vector': True}),
        ('preload', {'enable_preload': True}),
        ('workspace-double', {'set_workspace_multibuffer': 2}),
        ('ub-unlimited', {'limit_auto_multi_buffer_of_local_buffer': 'no-limit'}),
        ('bind-off', {'enable_auto_bind_sub_block': False}),
    ]
    rows = []
    torch.manual_seed(20261010)
    for dtype in (torch.float16, torch.float32, torch.bfloat16):
        for n in (1024, 2048, 4096):
            a = torch.randn((n, n), device='npu', dtype=dtype)
            b = torch.randn_like(a)
            bias = torch.randn((n,), device='npu', dtype=dtype)
            expected = baseline.run(a, b, bias)
            out = torch.empty_like(expected)
            reference = torch.relu(a @ b + bias)
            # 正式 FlagGems 门禁另行执行；此处同时保存与旧候选及 PyTorch 的差异。
            timing = do_bench_npu(lambda: baseline.run(a, b, bias), warmup=3, active=8)
            baseline_ms = float(timing[0] if isinstance(timing, list) else timing)
            for column_major in (False, True):
                for name, changes in variants:
                    options = {**base_options, **changes}
                    row = dict(dtype=str(dtype), n=n, name=name, config=[128, 128, 256],
                               column_major=column_major, options=options,
                               baseline_device_kernel_ms=baseline_ms)
                    try:
                        def launch():
                            mba_even_pipeline_kernel[((n // 128) ** 2,)](
                                a, b, bias, out, n, 128, 128, 256, column_major, **options)
                        launch()
                        torch.npu.synchronize()
                        row['max_abs_vs_parent'] = (out.float() - expected.float()).abs().max().item()
                        row['max_abs_vs_pytorch'] = (out.float() - reference.float()).abs().max().item()
                        if row['max_abs_vs_parent'] != 0:
                            raise ValueError('与完整通过的父候选存在数值差异')
                        timing = do_bench_npu(launch, warmup=3, active=8)
                        row['device_kernel_ms'] = float(timing[0] if isinstance(timing, list) else timing)
                    except Exception as error:
                        row['error'] = str(error)[-1800:]
                    rows.append(row)
                    artifact.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
                    print(json.dumps(row, ensure_ascii=False), flush=True)
            del a, b, bias, expected, out, reference


if __name__ == '__main__':
    main()
