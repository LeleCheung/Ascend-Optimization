#!/usr/bin/env python3
"""同一编译产物、同stream对比Tensor/实际地址启动；仅诊断，不计正式分数。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != '7':
        raise ValueError('输出存在或设备范围不符')
    import torch
    import torch_npu
    s = importlib.util.spec_from_file_location('launch_probe_candidate', a.source)
    module = importlib.util.module_from_spec(s)
    s.loader.exec_module(module)
    x = torch.full((1024,), 1.25, device='npu:0', dtype=torch.float32)
    compiled = module._log10_nomask.warmup(x, BLOCK=1024, grid=(1,))
    call = compiled.run.launch
    stream = torch_npu._C._npu_getCurrentRawStreamNoWait
    rows = []
    # 交错顺序检查，不把单次噪声当作机制收益。
    for kind in ['tensor', 'pointer', 'pointer', 'tensor']:
        samples = []
        for _ in range(3):
            x.fill_(1.25)
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(1000):
                value = x if kind == 'tensor' else x.data_ptr()
                call(1, 1, 1, stream(0), compiled.function, compiled.packed_metadata,
                     None, None, None, value, None)
            torch.npu.synchronize()
            samples.append((time.perf_counter()-start)*1e6/1000)
        rows.append(dict(argument=kind, walltime_us=samples, median_us=sorted(samples)[1]))
    data = dict(source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
        same_compiled_kernel=True, current_stream_each_call=True, rows=rows,
        scope='诊断探针；相同kernel、参数值、stream接口，改变传入Tensor或其当前data_ptr；不是KGS分数')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(data, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
