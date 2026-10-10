#!/usr/bin/env python3
"""诊断 PyTorch 连续 narrow_copy 的 kernel/内存复制事件与计时边界。"""
import json
import os
import csv
import shutil
import tarfile
import time
from pathlib import Path

import torch
import torch_npu
from triton.backends.ascend.testing import do_bench_npu_profiler


def main():
    root = Path(os.environ['KGS_DEBUG_ARTIFACTS'])
    rows = []
    for dtype in (torch.float16, torch.float32, torch.bfloat16):
        for shape in ((64, 64), (1024, 65536)):
            x = torch.randn(shape, device='npu', dtype=dtype)
            start, length = shape[0] // 4, shape[0] // 2
            fn = lambda: torch.narrow_copy(x, 0, start, length)
            for _ in range(10):
                fn()
            torch.npu.synchronize()
            row = {'dtype': str(dtype), 'shape': shape}
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU],
                record_shapes=True,
            ) as profiler:
                for _ in range(5):
                    fn()
                torch.npu.synchronize()
            name = f'{str(dtype).split(".")[-1]}-{shape[0]}-{shape[1]}.json'
            profiler.export_chrome_trace(str(root / name))
            trace = json.loads((root / name).read_text())
            events = trace if isinstance(trace, list) else trace.get('traceEvents', [])
            row['device_events'] = [
                {key: event.get(key) for key in ('name', 'cat', 'dur', 'args')}
                for event in events
                if 'kernel' in str(event.get('cat', '')).lower()
                or any(word in str(event.get('name', '')).lower()
                       for word in ('memcpy', 'memmove', 'narrow'))
            ]
            try:
                raw = root / ('raw-' + name.removesuffix('.json'))
                value = do_bench_npu_profiler(fn, warmup=3, active=8,
                                            prof_dir=str(raw), keep_res=True)
                row['do_bench_npu_ms'] = value
            except Exception as error:
                row['do_bench_npu_error'] = repr(error)
            row['task_csv'] = []
            for path in raw.rglob('task_time*.csv'):
                with path.open(newline='') as stream:
                    records = list(csv.DictReader(stream))
                row['task_csv'].append({'path': path.relative_to(raw).as_posix(), 'rows': records})
            with tarfile.open(str(raw) + '.tar.gz', 'w:gz') as archive:
                archive.add(raw, arcname=raw.name)
            shutil.rmtree(raw)
            begin = time.perf_counter_ns()
            for _ in range(100):
                fn()
            torch.npu.synchronize()
            row['synchronized_batch_us'] = (time.perf_counter_ns() - begin) / 100000
            rows.append(row)
            (root / 'pytorch-copy-timing.json').write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
