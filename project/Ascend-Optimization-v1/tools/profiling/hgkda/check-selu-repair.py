#!/usr/bin/env python3
"""SELU额外候选的原地别名、跨rank/stride/offset和特殊值检查，不参与性能计分。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def check(source, expected, output):
    if output.exists():
        raise ValueError('输出已存在；使用新目录保留原始证据')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != expected or os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != '7':
        raise ValueError('源码身份不同或未限定物理卡7')
    import torch
    import torch_npu  # noqa: F401
    assert torch.npu.device_count() == 1
    spec = importlib.util.spec_from_file_location('selu_repaired_candidate', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shapes = [(), (0,), (1,), (1025,), (8193,), (7, 9), (2, 3, 5),
              (2, 3, 5, 7), (2, 3, 2, 5, 7), (2, 2, 3, 2, 5, 7), (17,), (1024, 1025)]
    rows = []
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        for case, shape in enumerate(shapes):
            row = {'dtype': str(dtype), 'case': case, 'shape': list(shape), 'status': 'failed'}
            try:
                generator = torch.Generator().manual_seed(621001+case)
                original = torch.randn(shape, generator=generator).to(dtype)
                if case == 10:
                    original[:8] = torch.tensor([float('nan'), float('inf'), -float('inf'),
                        0., -0., -20., 20., .001], dtype=dtype)
                # 带offset的物理底座；一半case使用stride2并保留未使用lane哨兵。
                step = 2 if case % 2 else 1
                base = torch.full((original.numel()*step+4,), 19., dtype=dtype, device='npu:0')
                value = base[2:2+original.numel()*step:step].view(shape)
                value.copy_(original.to('npu:0'))
                if len(shape) >= 3:
                    value = value.permute(tuple(reversed(range(len(shape)))))
                before = value.cpu().clone()
                expected_value = torch.selu(before)
                row.update(input_stride=list(value.stride()), storage_offset=value.storage_offset())
                for repetition in range(2):
                    value.copy_(before.to('npu:0'))
                    actual = module.run(value)
                    torch.npu.synchronize()
                    assert actual is value and actual.shape == value.shape and actual.stride() == value.stride()
                    atol, rtol = ((2e-5, 2e-5) if dtype == torch.float32 else (.015, .015))
                    torch.testing.assert_close(actual.cpu(), expected_value, atol=atol, rtol=rtol, equal_nan=True)
                    assert bool((base[:2] == 19.).all()) and bool((base[-2:] == 19.).all())
                    if step == 2:
                        assert bool((base[3:2+original.numel()*2:2] == 19.).all())
                row.update(status='passed', repetitions=2)
            except Exception as error:
                row['error'] = type(error).__name__ + ': ' + str(error)
            rows.append(row)
    data = {'operator': 'selu_', 'source_sha256': digest,
            'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'physical_device': 7, 'logical_device': 'npu:0', 'total': len(rows),
            'passed': sum(row['status'] == 'passed' for row in rows), 'cases': rows,
            'scope': '36项额外原地/stride/offset/高rank/特殊值检查，每项两次；不替代KGS正式36项合同'}
    data['status'] = 'passed' if data['passed'] == data['total'] == 36 else 'failed'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((json.dumps(data, ensure_ascii=False, indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in data.items() if k != 'cases'}, ensure_ascii=False))
    return int(data['status'] != 'passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(check(args.source, args.expected_sha256, args.output))
