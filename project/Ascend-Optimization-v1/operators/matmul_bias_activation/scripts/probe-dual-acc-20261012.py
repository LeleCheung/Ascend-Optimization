#!/usr/bin/env python3
"""双 N 分块直接累计到矩阵累加器，验证数据复用与流水配置。"""
import importlib.util
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.backends.ascend.testing import do_bench_npu


@triton.jit
def mba_dual_acc_kernel(A, B, Bias, C, M: tl.constexpr, N: tl.constexpr,
                        K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                        BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    nm = M // BM
    nn = N // (2 * BN)
    if GROUP == 0:
        pm, pn = pid % nm, pid // nm
    else:
        width = GROUP * nn
        first = (pid // width) * GROUP
        size = tl.minimum(nm - first, GROUP)
        pm = first + pid % size
        pn = (pid % width) // size
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * (2 * BN) + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = A + rm[:, None] * K + rk[None, :]
    bp = B + rk[:, None] * N + rn[None, :]
    acc0 = tl.zeros((BM, BN), tl.float32)
    acc1 = tl.zeros((BM, BN), tl.float32)
    for _ in range(K // BK):
        a = tl.load(ap)
        b0 = tl.load(bp)
        b1 = tl.load(bp + BN)
        acc0 = tl.dot(a, b0, acc0, allow_tf32=False)
        acc1 = tl.dot(a, b1, acc1, allow_tf32=False)
        ap += BK
        bp += BK * N
    bias0 = tl.load(Bias + rn)
    bias1 = tl.load(Bias + rn + BN)
    acc0 += bias0[None, :]
    acc1 += bias1[None, :]
    acc0 = tl.maximum(acc0, acc0 * 0.0)
    acc1 = tl.maximum(acc1, acc1 * 0.0)
    cp = C + rm[:, None] * N + rn[None, :]
    tl.store(cp, acc0.to(bias0.dtype))
    tl.store(cp + BN, acc1.to(bias1.dtype))


def main():
    spec = importlib.util.spec_from_file_location('v8', 'profiling-dual-dispatch-v8-20261010.py')
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    artifacts = Path(os.environ['KGS_DEBUG_ARTIFACTS'])
    # 聚焦已知短板；各配置先编译和数值检查，再做设备计时。
    configs = [
        (128, 128, 256, 0, {}),
        (128, 128, 256, 0, {'multibuffer': True, 'unit_flag': True, 'sync_solver': False}),
        (128, 128, 128, 0, {'multibuffer': True, 'unit_flag': True, 'sync_solver': False}),
        (128, 64, 256, 0, {'multibuffer': True, 'unit_flag': True, 'sync_solver': False}),
        (64, 128, 256, 0, {'multibuffer': True, 'unit_flag': True, 'sync_solver': False}),
        (128, 128, 256, 4, {}),
        (128, 128, 512, 0, {}),
    ]
    torch.manual_seed(20261012)
    rows = []
    for dtype in (torch.float16, torch.bfloat16):
        for n in (1024, 2048, 4096):
            a = torch.randn((n, n), device='npu', dtype=dtype)
            b = torch.randn_like(a)
            bias = torch.randn((n,), device='npu', dtype=dtype)
            expected = baseline.run(a, b, bias)
            reference = torch.relu(a @ b + bias)
            out = torch.empty_like(expected)
            value = do_bench_npu(lambda: baseline.run(a, b, bias), warmup=5, active=15)
            baseline_ms = float(value[0] if isinstance(value, list) else value)
            for bm, bn, bk, group, options in configs:
                row = dict(dtype=str(dtype), n=n, tile=[bm, bn, bk], group=group,
                           options=options, baseline_ms=baseline_ms)
                try:
                    def launch():
                        mba_dual_acc_kernel[((n // bm) * (n // (2 * bn)),)](
                            a, b, bias, out, n, n, n, bm, bn, bk, group, **options)
                    launch()
                    torch.npu.synchronize()
                    row['max_abs_vs_v8'] = (out.float() - expected.float()).abs().max().item()
                    row['max_abs_vs_pytorch'] = (out.float() - reference.float()).abs().max().item()
                    if not torch.isfinite(out).all().item():
                        raise ValueError('出现非有限输出')
                    value = do_bench_npu(launch, warmup=5, active=15)
                    row['device_kernel_ms'] = float(value[0] if isinstance(value, list) else value)
                    row['speedup_vs_v8'] = baseline_ms / row['device_kernel_ms']
                except Exception as error:
                    row['error'] = str(error)[-1800:]
                rows.append(row)
                (artifacts / 'dual-acc.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
                print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
