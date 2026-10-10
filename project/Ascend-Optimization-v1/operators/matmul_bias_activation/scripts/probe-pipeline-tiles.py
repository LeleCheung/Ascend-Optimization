#!/usr/bin/env python3
"""诊断候选的分块与 dot 累加；事件计时只用于筛选，最终须完整 evaluate。"""
import importlib.util
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def tuned_pipeline(A, B, Bias, C, M: tl.constexpr, N: tl.constexpr,
                   K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                   BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    if GROUP == 0:
        pm, pn = pid // nn, pid % nn
    else:
        width = GROUP * nn
        first = (pid // width) * GROUP
        size = tl.minimum(nm - first, GROUP)
        pm = first + pid % size
        pn = (pid % width) // size
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = A + rm[:, None] * K + rk[None, :]
    bp = B + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        a = tl.load(ap, (rm[:, None] < M) & (rk[None, :] + start * BK < K), 0)
        b = tl.load(bp, (rk[:, None] + start * BK < K) & (rn[None, :] < N), 0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        ap += BK
        bp += BK * N
    acc += tl.load(Bias + rn, rn < N, 0)[None, :]
    acc = tl.maximum(acc, acc * 0.0)
    tl.store(C + rm[:, None] * N + rn[None, :], acc,
             (rm[:, None] < M) & (rn[None, :] < N))


def main():
    output = Path(os.environ['KGS_DEBUG_ARTIFACTS']) / 'tile-results.json'
    source = Path('profiling-k128-pipeline-v3-20261010.py')
    spec = importlib.util.spec_from_file_location('v3', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    configs = [(128,128,128,0), (128,128,256,0), (128,256,128,0),
               (256,128,128,0), (128,128,128,8), (128,256,128,8)]
    rows = []
    torch.manual_seed(20261010)
    for dtype in [torch.float16, torch.float32, torch.bfloat16]:
        for n in [1024, 4096]:
            a = torch.randn((n,n), device='npu', dtype=dtype)
            b = torch.randn_like(a)
            bias = torch.randn((n,), device='npu', dtype=dtype)
            expected = module.run(a,b,bias)
            for config in configs:
                row = {'dtype':str(dtype), 'n':n, 'config':config}
                try:
                    bm,bn,bk,group = config
                    out = torch.empty_like(expected)
                    def launch():
                        tuned_pipeline[(triton.cdiv(n,bm)*triton.cdiv(n,bn),)](
                            a,b,bias,out,n,n,n,bm,bn,bk,group,
                            multibuffer=True, unit_flag=True, sync_solver=False)
                    launch()
                    torch.npu.synchronize()
                    # 完整数值验收由正式框架负责；此处保存与已通过候选的差异。
                    row['max_abs_vs_v3'] = (out.float()-expected.float()).abs().max().item()
                    measurements = []
                    for _ in range(3):
                        start = torch.npu.Event(enable_timing=True)
                        end = torch.npu.Event(enable_timing=True)
                        start.record()
                        for _ in range(10): launch()
                        end.record()
                        torch.npu.synchronize()
                        measurements.append(start.elapsed_time(end)/10)
                    row['event_ms'] = measurements
                except Exception as error:
                    row['error'] = str(error)[:1000]
                rows.append(row)
                output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
                print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
