#!/usr/bin/env python3
"""按真实布局筛选归约分块；事件计时用于筛选，完整框架负责最终验收。"""
import importlib.util
import json
import os
from pathlib import Path
import statistics
import torch
import triton


def main():
    spec = importlib.util.spec_from_file_location('candidate', 'direct-axis-global-v3-20261010.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    artifact = Path(os.environ['KGS_DEBUG_ARTIFACTS']) / 'direct-tiles.json'
    rows = []
    torch.manual_seed(20261010)
    for dtype in (torch.float16, torch.float32, torch.bfloat16):
        for shape in ((4096,4096), (64,512,512), (1024,1024,1024)):
            x = torch.randn(shape, dtype=dtype, device='npu')
            expected = torch.amin(x, dim=1)
            out = torch.empty_like(expected)
            last = len(shape) == 2
            configs = ([(1,2048), (4,512), (8,512), (16,256), (16,512),
                        (32,256), (8,1024), (4,4096)] if last else
                       [(64,128), (128,32), (256,16), (256,32), (256,64),
                        (512,16), (512,32), (128,64)])
            for block, br in configs:
                row = {'dtype':str(dtype), 'shape':shape, 'config':[block,br]}
                try:
                    if last:
                        def launch():
                            mod.amin_direct_last_kernel[(triton.cdiv(shape[0],block),)](
                                x,out,shape[0],shape[1],block,br)
                    else:
                        def launch():
                            mod.amin_direct_middle_kernel[(shape[0],triton.cdiv(shape[2],block))](
                                x,out,shape[1],shape[2],block,br)
                    launch();torch.npu.synchronize()
                    row['equal'] = bool(torch.equal(out,expected))
                    times = []
                    for _ in range(3):
                        a = torch.npu.Event(enable_timing=True)
                        b = torch.npu.Event(enable_timing=True)
                        a.record()
                        for _ in range(5):launch()
                        b.record();torch.npu.synchronize()
                        times.append(a.elapsed_time(b)/5)
                    row['event_ms'] = times
                    row['median_ms'] = statistics.median(times)
                except Exception as error:
                    row['error'] = str(error)[:1500]
                rows.append(row)
                artifact.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
                print(json.dumps(row,ensure_ascii=False),flush=True)
            del x, expected, out


if __name__ == '__main__':
    main()
