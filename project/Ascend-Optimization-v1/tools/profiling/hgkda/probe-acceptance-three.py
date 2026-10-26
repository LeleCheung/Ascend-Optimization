#!/usr/bin/env python3
"""剩余三项验收的隔离进程诊断；补充NPU分解路径不替换原参考。"""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True

ROOT = Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1')
BATCH = ROOT / 'operators/campaigns/acceptance-three-20261026'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + chr(10))


def profile(label, function, output):
    import torch
    import torch_npu
    folder = output / label
    folder.mkdir()
    function()
    torch.npu.synchronize()
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU], record_shapes=True, with_stack=False) as prof:
        with torch.profiler.record_function('HGKDA_ACCEPTANCE_' + label):
            function()
            torch.npu.synchronize()
    prof.export_chrome_trace(str(folder / 'trace.json'))
    data = json.loads((folder / 'trace.json').read_text())
    events = data if isinstance(data, list) else data.get('traceEvents', [])
    from collections import Counter
    names = Counter(e.get('name') for e in events if e.get('ph') == 'X')
    transfers = [{k: e[k] for k in ['name', 'cat', 'dur', 'args'] if k in e}
                 for e in events if e.get('ph') == 'X' and any(s in str(e.get('name', '')).lower()
                 for s in ['memcpy', 'fallback'])]
    summary = dict(events=dict(names), transfers=transfers)
    save(folder / 'summary.json', summary)
    return summary


def latency(function, repetitions=11):
    import torch
    for _ in range(3):
        function()
    torch.npu.synchronize()
    times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        result = function()
        torch.npu.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return dict(median_ms=sorted(times)[len(times)//2], samples_ms=times)


def cumulative(output):
    import torch
    import numpy as np
    old = ROOT / 'operators/campaigns/two-closure-20261024'
    source = old / 'deliverables/reviewed/cumulative_trapezoid/main.py'
    candidate = load('scan_original', source)
    # 已安装头文件明确给出的V2接口；只在新隔离进程中调用。
    variant = output / 'cumsum-v2.py'
    text = source.read_text().replace(
        'lib.aclnnCumsumGetWorkspaceSize.argtypes = [ct.c_void_p, ct.c_int64,\n        ct.c_int, ct.c_void_p,',
        'lib.aclnnCumsumV2GetWorkspaceSize.argtypes = [ct.c_void_p, ct.c_int64,\n        ct.c_bool, ct.c_bool, ct.c_void_p,')
    text = text.replace('aclnnCumsumGetWorkspaceSize', 'aclnnCumsumV2GetWorkspaceSize')
    text = text.replace('aclnnCumsum.argtypes', 'aclnnCumsumV2.argtypes').replace('aclnnCumsum.restype', 'aclnnCumsumV2.restype')
    text = text.replace('lib.aclnnCumsum(', 'lib.aclnnCumsumV2(').replace('desc_in, axis, dtype, desc_out,', 'desc_in, axis, False, False, desc_out,')
    variant.write_text(text)
    v2 = load('scan_v2', variant)
    inputs = old / 'research/reference-stability/default-256x8-cumulative_trapezoid-input.bin'
    assert inputs.exists()
    y = torch.from_numpy(np.frombuffer(inputs.read_bytes(), dtype=np.float16).copy().reshape(256, 8)).to('npu:0')
    rows = []
    for deterministic in [False, True]:
        torch.use_deterministic_algorithms(deterministic)
        for label, fn in [('candidate', lambda: candidate.run(y, dim=0)),
                          ('candidate_v2', lambda: v2.run(y, dim=0)),
                          ('reference', lambda: torch.cumulative_trapezoid(y, dim=0))]:
            row = dict(label=label, deterministic=deterministic)
            try:
                values = []
                for _ in range(8):
                    got = fn()
                    torch.npu.synchronize()
                    values.append(got.cpu().float().numpy())
                stack = np.stack(values)
                radius = .015 + .015 * np.abs(stack)
                row.update(status='completed', bitwise_stable=all(np.array_equal(values[0], x) for x in values[1:]),
                           empty_tolerance_intersections=int((np.max(stack-radius, axis=0)>np.min(stack+radius, axis=0)).sum()),
                           max_repeat_difference=float(np.max(stack.max(0)-stack.min(0))),
                           latency=latency(fn), profile=profile(label + ('_det' if deterministic else '_default'), fn, output))
            except Exception as error:
                row.update(status='failed', error=type(error).__name__ + ': ' + str(error)[-4000:])
            rows.append(row)
            save(output / 'results.json', dict(scope='同固定输入、隔离进程官方接口诊断，不作完整性能分数',
                 input_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(), rows=rows))
            print(json.dumps({k:v for k,v in row.items() if k!='profile'}), flush=True)
    torch.use_deterministic_algorithms(False)


def conj_reference(input):
    """纯NPU等价分解：实数复制；复数通过实数视图复制并翻转虚部。"""
    import torch
    if not input.is_complex():
        return input.contiguous().clone()
    raw = input.conj() if input.is_conj() else input
    out = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    src = torch.view_as_real(raw)
    dst = torch.view_as_real(out)
    dst.copy_(src)
    if not input.is_conj():
        dst[..., 1].neg_()
    return out


def conjugate(output):
    import torch
    parent = ROOT / 'operators/campaigns/remaining-five-20261023-v2'
    candidate = load('conj_candidate', parent / 'deliverables/reviewed/_conj_physical/main.py')
    oracle = load('conj_oracle', parent / 'catalog/ops/_conj_physical/oracle.py')
    cases = [json.loads(l) for l in (parent/'catalog/ops/_conj_physical/timing.jsonl').read_text().splitlines()]
    rows = []
    for index, case in enumerate(cases):
        args = oracle.gen_inputs(case, 'npu:0')
        expected = oracle.run(**args)
        ref = conj_reference(**args)
        got = candidate.run(**args)
        torch.npu.synchronize()
        torch.testing.assert_close(ref.cpu(), expected.cpu(), atol=0, rtol=0)
        torch.testing.assert_close(got.cpu(), expected.cpu(), atol=0, rtol=0)
        row = dict(case=case['name'], status='passed', current=latency(lambda: candidate.run(**args)),
                   npu_decomposition=latency(lambda: conj_reference(**args)))
        if index in [0, 3]:
            row['default_profile'] = profile('default_' + str(index), lambda: oracle.run(**args), output)
            row['npu_profile'] = profile('npu_' + str(index), lambda: conj_reference(**args), output)
        rows.append(row)
        save(output/'results.json', dict(scope='纯NPU等价分解补充对照，原始PyTorch默认参考不变', rows=rows))
        print(json.dumps({k:v for k,v in row.items() if 'profile' not in k}), flush=True)
    extras = []
    for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.complex64]:
        for layout in ['transpose', 'stride2', 'offset', 'conj', 'empty']:
            x = torch.randn((5,14), dtype=dtype).to('npu:0')
            x = x.t() if layout=='transpose' else x[:,::2] if layout=='stride2' else x[1:,1:10] if layout=='offset' else x.conj() if layout=='conj' else x[:0]
            expected = torch.ops.aten._conj_physical(x.cpu())
            got = conj_reference(x)
            torch.npu.synchronize()
            torch.testing.assert_close(got.cpu(), expected, atol=0, rtol=0)
            assert not got.is_conj() and (not x.numel() or got.data_ptr()!=x.data_ptr())
            extras.append(dict(dtype=str(dtype), layout=layout, status='passed'))
    save(output/'supplementary.json', dict(status='passed', total=len(extras), cases=extras))


def main():
    import torch
    import torch_npu
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == '7' and torch.npu.device_count() == 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['cumulative', 'conjugate'], required=True)
    args = parser.parse_args()
    output = BATCH/'research'/args.kind
    output.mkdir(parents=True, exist_ok=False)
    (cumulative if args.kind=='cumulative' else conjugate)(output)


if __name__ == '__main__':
    main()
