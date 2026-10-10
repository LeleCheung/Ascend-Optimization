#!/usr/bin/env python3
"""采集混合候选的小/大复制 trace，核验实际 DMA/kernel 和输出语义。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import torch
import torch_npu


def device_events(trace):
    """只提取设备任务；部分 torch_npu trace 不提供 cat。"""
    events = trace if isinstance(trace, list) else trace.get('traceEvents', [])
    result = []
    for event in events:
        args = event.get('args') or {}
        identified = bool(args.get('Task Type')) and (
            'Task Id' in args or 'Physic Stream Id' in args)
        if event.get('ph') == 'X' and (identified or
                'kernel' in str(event.get('cat', '')).lower()):
            result.append(event)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidates', nargs='*', default=[
        'small-dma-hybrid-20261010.py', 'contiguous-dma-hybrid-20261010.py'])
    args = parser.parse_args()
    root = Path(os.environ['KGS_DEBUG_ARTIFACTS'])
    rows = []
    for filename in args.candidates:
        source = Path(filename)
        spec = importlib.util.spec_from_file_location('hybrid_candidate', source)
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        for dtype in (torch.float16, torch.float32, torch.bfloat16):
            for shape in ((64, 64), (1024, 65536)):
                x = torch.randn(shape, device='npu', dtype=dtype)
                start, length = shape[0] // 4, shape[0] // 2
                expected = torch.narrow_copy(x, 0, start, length)
                actual = candidate.run(x, 0, start, length)
                torch.npu.synchronize()
                assert torch.equal(expected, actual) and actual.data_ptr() != x.data_ptr()
                for _ in range(3):
                    candidate.run(x, 0, start, length)
                torch.npu.synchronize()
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU,
                                torch_npu.profiler.ProfilerActivity.NPU], record_shapes=True,
                ) as profiler:
                    for _ in range(3):
                        candidate.run(x, 0, start, length)
                    torch.npu.synchronize()
                trace_name = source.stem + '-' + str(dtype).split('.')[-1] + '-' + str(shape[0]) + '.json'
                profiler.export_chrome_trace(str(root / trace_name))
                trace = json.loads((root / trace_name).read_text())
                device = device_events(trace)
                row = {'source': filename, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                       'dtype': str(dtype), 'shape': shape, 'dim': 0, 'start': start, 'length': length,
                       'equal': True, 'new_output': True, 'trace': trace_name,
                       'trace_sha256': hashlib.sha256((root / trace_name).read_bytes()).hexdigest(),
                       'events': [{key: event.get(key) for key in ('name', 'cat', 'dur', 'args')}
                                  for event in device]}
                rows.append(row)
                (root / 'hybrid-copy-trace.json').write_text(
                    json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
                print(filename, str(dtype), shape, 'TRACE_READY', len(device), flush=True)
                del x, expected, actual


if __name__ == '__main__':
    main()
