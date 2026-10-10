#!/usr/bin/env python3
"""先同步单次大输入启动，定位 master 网格与快路径的运行差异。"""
import importlib.util
import json
import os
import time
from pathlib import Path

import torch


def main():
    rows = []
    output = Path(os.environ['KGS_DEBUG_ARTIFACTS']) / 'large-launch.json'
    for name in ('contiguous-copy-20261010.py', 'flaggems-master-grid-compatible-20261010.py'):
        spec = importlib.util.spec_from_file_location('candidate_' + str(len(rows)), name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for dtype in (torch.float16, torch.float32, torch.bfloat16):
            x = torch.randn((1024, 65536), device='npu', dtype=dtype)
            expected = torch.narrow_copy(x, 0, 256, 512)
            torch.npu.synchronize()
            print('BEFORE', name, str(dtype), flush=True)
            begin = time.perf_counter()
            out = module.run(x, 0, 256, 512)
            print('ENQUEUED', name, str(dtype), flush=True)
            torch.npu.synchronize()
            equal = torch.equal(expected, out)
            assert equal
            row = {'candidate': name, 'dtype': str(dtype), 'equal': equal,
                   'cold_synchronized_seconds': time.perf_counter() - begin}
            rows.append(row)
            output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
            print('DONE', row, flush=True)
            del x, expected, out


if __name__ == '__main__':
    main()
