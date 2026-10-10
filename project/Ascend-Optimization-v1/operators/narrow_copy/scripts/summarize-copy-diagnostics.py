#!/usr/bin/env python3
"""核验独立真机采样，生成设备任务与调用耗时的中文对照。"""
import argparse
import hashlib
import json
import math
from pathlib import Path


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


def verified_job(root):
    response = json.loads((root / 'response.json').read_text(encoding='utf-8'))
    assert response.get('status') == 'SUCCEEDED', '诊断任务未成功：' + str(root)
    for item in response.get('artifacts', []):
        path = (root / 'artifacts' / item['path']).resolve()
        path.relative_to((root / 'artifacts').resolve())
        data = path.read_bytes()
        assert len(data) == item['size_bytes']
        assert hashlib.sha256(data).hexdigest() == item['sha256']


def number(value):
    assert type(value) in (int, float) and math.isfinite(value) and value > 0
    return f'{value:.2f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    operator = project / 'operators/narrow_copy'
    diagnostics = operator / 'reports/diagnostics-20261010'
    scopes = diagnostics / 'narrow-current-device-task-scopes-20261010'
    trace = diagnostics / 'narrow-hybrid-copy-trace-20261010'
    verified_job(scopes)
    verified_job(trace)
    timings = json.loads((scopes / 'artifacts/timing-scopes.json').read_text(encoding='utf-8'))
    pure = operator / 'candidates/contiguous-copy-20261010.py'
    assert timings['source_sha256'] == hashlib.sha256(pure.read_bytes()).hexdigest()
    assert timings['device_scope'] == 'device_task'
    assert len(timings['cases']) == 6
    observed = set()
    lines = ['# narrow_copy 真机耗时与复制路径', '',
             '设备任务时间与独立 host 采样分别记录；同步调用时间包含发射和等待。下表单位均为 μs，host 列取 20 次有效样本的中位数。', '',
             '| dtype | 输入形状 | Triton 设备任务 | PyTorch 设备任务 | 预分配 JIT 设备任务 | Triton 发射 | PyTorch 发射 | Triton 同步调用 | PyTorch 同步调用 | 分配 |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in timings['cases']:
        key = (row['dtype'], tuple(row['shape']))
        assert key not in observed and row['equal'] is True
        observed.add(key)
        host = row['host']
        values = [row[name + '_device_task_us'] for name in ('candidate', 'pytorch', 'allocated_jit_launch')]
        values += [host[name]['enqueue_us']['median_us'] for name in ('candidate', 'pytorch')]
        values += [host[name]['synchronized_call_us']['median_us'] for name in ('candidate', 'pytorch')]
        values += [host['allocation']['enqueue_us']['median_us']]
        lines.append('| ' + row['dtype'].removeprefix('torch.') + ' | ' + str(row['shape']) +
                     ' | ' + ' | '.join(number(v) for v in values) + ' |')
    assert observed == {(dt, shape) for dt in ('torch.float16', 'torch.float32', 'torch.bfloat16')
                        for shape in ((64, 64), (1024, 65536))}
    ratios = {}
    for shape in ((64, 64), (1024, 65536)):
        rows = [row for row in timings['cases'] if tuple(row['shape']) == shape]
        ratios[shape] = math.exp(sum(math.log(row['candidate_device_task_us'] /
                                             row['pytorch_device_task_us']) for row in rows) / len(rows))
    lines += ['', '## 优化判断', '',
              f"三种 dtype 的小输入中，Triton 设备任务耗时为 PyTorch 的 {ratios[(64, 64)]:.2f} 倍；"
              f"大输入为 {ratios[(1024, 65536)]:.2f} 倍（均为三种 dtype 比值的几何平均）。",
              '小输入优先测试连续片段的 CANN DMA 路径；大输入重点检查复制分块、program 数和索引计算。'
              '这两类输入需要分别处理，不能从 host 发射耗时推断所有 case 的设备瓶颈。', '']
    seen = set()
    lines += ['', '## 混合候选实际设备任务', '',
              '候选均重新分配输出并验证逐值相等。任务名称来自真机 Chrome trace。', '',
              '| 候选 | dtype | 输入形状 | 实际任务名称 |', '| --- | --- | --- | --- |']
    jobs = [(trace, {'small-dma-hybrid-20261010.py', 'contiguous-dma-hybrid-20261010.py'})]
    extra = diagnostics / 'narrow-profile-dma-trace-20261010'
    if (extra / 'response.json').exists():
        verified_job(extra)
        jobs.append((extra, {'profile-small-dma-hybrid-20261010.py'}))
    for trace_root, sources in jobs:
        rows = json.loads((trace_root / 'artifacts/hybrid-copy-trace.json').read_text(encoding='utf-8'))
        assert len(rows) == 6 * len(sources)
        expected = {(source, dtype, shape) for source in sources
                    for dtype in ('torch.float16', 'torch.float32', 'torch.bfloat16')
                    for shape in ((64, 64), (1024, 65536))}
        actual = set()
        for row in rows:
            append_trace_row(row, trace_root, operator, seen, actual, lines)
        assert actual == expected, '设备 trace 的候选、dtype 或形状不完整'
    lines += ['', '设备任务计时排除 host 发射间隙，小输入的 DMA 与 Triton 差距因此不能只归因于 Python。'
              '同步调用与设备任务来自不同采样，不能直接相减得到精确 host 开销。复制不执行矩阵计算，'
              '本报告用任务类型、实际耗时和布局分析解释性能，没有计算数值 Roofline 或声称已达到硬件峰值。', '']
    output = operator / 'reports/master-closure-20261010/真机诊断.md'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines), encoding='utf-8')
    print('COPY_DIAGNOSTICS_VERIFIED', output)


def append_trace_row(row, trace, operator, seen, actual, lines):
    assert row['equal'] is True and row['new_output'] is True
    source = operator / 'candidates' / row['source']
    assert row['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    path = (trace / 'artifacts' / row['trace']).resolve()
    path.relative_to((trace / 'artifacts').resolve())
    assert row['trace_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    key = (row['source'], row['dtype'], tuple(row['shape']))
    assert key not in seen
    seen.add(key)
    actual.add(key)
    raw_trace = json.loads(path.read_text(encoding='utf-8'))
    names = sorted({event['name'] for event in device_events(raw_trace) if event.get('name')})
    assert names, '没有可核查的设备事件'
    lines.append('| ' + row['source'] + ' | ' + row['dtype'].removeprefix('torch.') +
                 ' | ' + str(row['shape']) + ' | ' + ', '.join(names).replace('|', '/') + ' |')

if __name__ == '__main__':
    main()
