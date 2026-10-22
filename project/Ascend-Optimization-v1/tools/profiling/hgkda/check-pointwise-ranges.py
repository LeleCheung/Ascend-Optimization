#!/usr/bin/env python3
"""验证输入dtype的完整指数范围；防止BF16经FP16临时量截断大正值。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator', choices=['log10_', 'selu_'], required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != '7':
        raise ValueError('输出存在或设备范围不同')
    import torch
    import torch_npu  # noqa: F401
    spec = importlib.util.spec_from_file_location('range_candidate', args.source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        info = torch.finfo(dtype)
        values = torch.tensor([float('nan'), float('inf'), -float('inf'), 0., -0.,
            info.max/4, -info.max/4, info.tiny, -info.tiny, 1e5, -1e5, 1e20, -1e20,
            1e-20, -1e-20, .0001, -.0001, .5, -.5, 1., -1., 10., -10., 100., -100.], dtype=dtype)
        for n in [1024, 1048576, 4194304]:
            row = dict(dtype=str(dtype), numel=n, status='failed')
            try:
                original = values.repeat((n+len(values)-1)//len(values))[:n].contiguous()
                expected = getattr(torch, args.operator[:-1])(original)
                value = original.to('npu:0')
                stream = torch.npu.Stream()
                for repetition in range(2):
                    with torch.npu.stream(stream if repetition else torch.npu.default_stream()):
                        value.copy_(original)
                        actual = module.run(value)
                    torch.npu.synchronize()
                    assert actual is value
                    tolerance = 1e-4 if dtype is torch.float32 else .015
                    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance,
                        rtol=tolerance, equal_nan=True)
                row.update(status='passed', repetitions=2, stream_switch_checked=True)
            except Exception as error:
                row['error'] = type(error).__name__+': '+str(error)
            rows.append(row)
    result = dict(operator=args.operator, cases=rows, total=9,
        passed=sum(r['status']=='passed' for r in rows), physical_device=7, logical_device='npu:0',
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='9项dtype数值范围检查，含大小输入、极值、特殊值与stream切换；独立于正式计分')
    result['status'] = 'passed' if result['passed'] == 9 else 'failed'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}, ensure_ascii=False), flush=True)
    raise SystemExit(result['status'] != 'passed')


if __name__ == '__main__':
    main()
