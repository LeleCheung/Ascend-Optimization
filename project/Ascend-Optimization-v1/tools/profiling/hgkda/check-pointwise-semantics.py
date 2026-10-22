#!/usr/bin/env python3
"""log10_/selu_的独立扩展语义检查；不修改KGS合同或计分。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def check(operator, source, output):
    if output.exists():
        raise ValueError('结果已存在，请使用新目录')
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != '7':
        raise ValueError('本实验只允许物理卡7')
    import torch
    import torch_npu  # noqa: F401
    assert torch.npu.device_count() == 1
    spec = importlib.util.spec_from_file_location('pointwise_candidate', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shapes = [(), (0,), (1,), (1025,), (8193,), (7, 9), (2, 3, 5),
              (2, 3, 5, 7), (2, 3, 2, 5, 7), (2, 2, 3, 2, 5, 7),
              (17,), (1024, 1025), (2, 2, 2, 2, 2, 2, 2, 2, 3), (4097,)]
    records = []
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        for case, shape in enumerate(shapes):
            row = dict(dtype=str(dtype), case=case, shape=list(shape), status='failed')
            try:
                generator = torch.Generator().manual_seed(622001 + case)
                original = torch.randn(shape, generator=generator)
                if operator == 'log10_':
                    original = original.abs() + .05
                original = original.to(dtype)
                if case == 10:
                    original[:12] = torch.tensor([float('nan'), float('inf'), -float('inf'),
                        0., -0., -1., 1., 1.001, .999, 1e-5, 1e3, .5], dtype=dtype)
                step = 2 if case % 2 or case == 12 else 1
                base = torch.full((original.numel() * step + 4,), 19., dtype=dtype, device='npu:0')
                value = base[2:2+original.numel()*step:step].view(shape)
                # 在低rank底座写入，避开CANN copy算子自身最多8维的限制。
                base[2:2+original.numel()*step:step].copy_(original.reshape(-1).to('npu:0'))
                if len(shape) >= 3:
                    value = value.permute(tuple(reversed(range(len(shape)))))
                axes = tuple(reversed(range(len(shape)))) if len(shape) >= 3 else tuple(range(len(shape)))
                before = original.permute(axes).clone() if len(shape) >= 3 else original.clone()
                reference = getattr(torch, operator[:-1])(before)
                row.update(stride=list(value.stride()), storage_offset=value.storage_offset())
                # 先在默认stream执行，再切换新stream，验证缓存没有固定旧stream。
                stream = torch.npu.Stream()
                for repetition in range(2):
                    context = torch.npu.stream(stream if repetition else torch.npu.default_stream())
                    with context:
                        base[2:2+original.numel()*step:step].copy_(original.reshape(-1).to('npu:0'))
                        actual = module.run(value)
                    torch.npu.synchronize()
                    assert actual is value and actual.stride() == value.stride()
                    atol, rtol = (1e-4, 1e-4) if dtype == torch.float32 else (.015, .015)
                    # 将底座降到1维传回CPU，再恢复真实stride布局；避免观测端rank限制。
                    observed = base.cpu()[2:2+original.numel()*step:step].view(shape)
                    if len(shape) >= 3:
                        observed = observed.permute(axes)
                    torch.testing.assert_close(observed, reference, atol=atol, rtol=rtol, equal_nan=True)
                    assert bool((base[:2] == 19.).all()) and bool((base[-2:] == 19.).all())
                    if step == 2:
                        assert bool((base[3:2+original.numel()*2:2] == 19.).all())
                row.update(status='passed', repetitions=2, stream_switch_checked=True)
            except Exception as error:
                row['error'] = type(error).__name__ + ': ' + str(error)
            records.append(row)
    data = dict(operator=operator, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                physical_device=7, logical_device='npu:0', cases=records,
                total=len(records), passed=sum(r['status'] == 'passed' for r in records),
                scope='独立语义检查：原地、offset、stride、高rank、特殊值、当前stream；不计入正式性能')
    data['status'] = 'passed' if data['passed'] == data['total'] else 'failed'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in data.items() if k != 'cases'}, ensure_ascii=False), flush=True)
    return int(data['status'] != 'passed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operator', choices=['log10_', 'selu_'], required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    raise SystemExit(check(a.operator, a.source, a.output))
