#!/usr/bin/env python3
"""成对OpProf驱动：同输入，一次目标调用；探针计时另存，不当作KGS成绩。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--side', choices=['candidate', 'reference'], required=True)
    p.add_argument('--probe', type=Path)
    a = p.parse_args()
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != '7':
        raise ValueError('只允许物理卡7')
    import torch
    import torch_npu  # noqa: F401
    assert torch.npu.device_count() == 1
    spec = json.loads(a.spec.read_text())
    dtype = getattr(torch, spec['dtype'])
    g = torch.Generator().manual_seed(spec['seed'])
    cpu = torch.randn(spec['shape'], generator=g)
    if spec['operator'] == 'log10_':
        cpu = cpu.abs() + .05
    value = cpu.to(dtype=dtype, device='npu:0')
    original = value.clone()
    if a.side == 'reference':
        run = getattr(torch, spec['operator'])
    else:
        source = Path(spec['source'])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == spec['source_sha256']
        modspec = importlib.util.spec_from_file_location('paired_candidate', source)
        module = importlib.util.module_from_spec(modspec)
        modspec.loader.exec_module(module)
        run = module.run
    torch.npu.synchronize()
    if not a.probe:
        run(value)
        torch.npu.synchronize()
        return
    if a.probe.exists():
        raise ValueError('探针输出已存在')
    # 本探针与KGS均采用同步前后包围的批量完整调用；不替代正式评测。
    # 原地操作每次都实际执行，数据会改变；正式合同同样批量调用。
    run(value)
    torch.npu.synchronize()
    samples = []
    for _ in range(3):
        value.copy_(original)
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(300):
            run(value)
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start)*1e6/300)
    a.probe.parent.mkdir(parents=True, exist_ok=True)
    a.probe.write_text(json.dumps(dict(side=a.side, spec=spec, walltime_us=samples,
        median_us=sorted(samples)[1], iterations=300,
        scope='独立批量完整调用探针；不是KGS正式成绩'), ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
