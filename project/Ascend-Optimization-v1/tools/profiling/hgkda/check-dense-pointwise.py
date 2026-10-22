#!/usr/bin/env python3
"""补充检查实际计分规模的连续对齐快路径，独立于既有42项语义记录。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operator', choices=['log10_', 'selu_'], required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != '7':
        raise ValueError('输出存在或物理卡范围不同')
    import torch
    import torch_npu  # noqa: F401
    s = importlib.util.spec_from_file_location('dense_pointwise_candidate', a.source)
    module = importlib.util.module_from_spec(s)
    s.loader.exec_module(module)
    records = []
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        for shape in [(1024,), (65536,), (1048576,), (4096,4096)]:
            row = dict(dtype=str(dtype), shape=list(shape), status='failed')
            try:
                g = torch.Generator().manual_seed(622419)
                before = torch.randn(shape, generator=g)
                if a.operator == 'log10_':
                    before = before.abs()+.05
                before = before.to(dtype)
                expected = getattr(torch, a.operator[:-1])(before)
                n = before.numel()
                base = torch.full((n+16,), 19., dtype=dtype, device='npu:0')
                value = base[8:8+n].view(shape)
                assert value.data_ptr() % 16 == 0
                stream = torch.npu.Stream()
                for repetition in range(2):
                    with torch.npu.stream(stream if repetition else torch.npu.default_stream()):
                        value.copy_(before.to('npu:0'))
                        actual = module.run(value)
                    torch.npu.synchronize()
                    assert actual is value
                    atol,rtol = (1e-4,1e-4) if dtype == torch.float32 else (.015,.015)
                    torch.testing.assert_close(actual.cpu(), expected, atol=atol, rtol=rtol, equal_nan=True)
                    assert bool((base[:8] == 19.).all()) and bool((base[-8:] == 19.).all())
                row.update(status='passed', repetitions=2, stream_switch_checked=True)
            except Exception as error:
                row['error'] = type(error).__name__+': '+str(error)
            records.append(row)
    result = dict(operator=a.operator, source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        physical_device=7,logical_device='npu:0',cases=records,total=len(records),
        passed=sum(r['status']=='passed' for r in records),
        scope='12项连续对齐快路径，覆盖所有计分规模、dtype和stream切换；每项两次，独立于正式计分')
    result['status'] = 'passed' if result['passed'] == result['total'] == 12 else 'failed'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False),flush=True)
    raise SystemExit(result['status']!='passed')


if __name__ == '__main__':
    main()
