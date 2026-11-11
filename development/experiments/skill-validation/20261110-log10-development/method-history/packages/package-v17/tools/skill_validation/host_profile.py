"""只读候选的热调用宿主栈诊断；独立于正式性能合同，不修改候选。"""
import argparse
import cProfile
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import pstats
import time


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['source', 'leaf', 'output']:
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--case', action='append', required=True)
    parser.add_argument('--iterations', type=int, default=50)
    args = parser.parse_args()
    assert not args.output.exists()
    assert 1 <= args.iterations <= 1000
    import torch
    import torch_npu
    import triton.backends.ascend.driver as driver
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    candidate = imported('_host_profile_candidate', args.source)
    oracle = imported('_host_profile_oracle', args.leaf/'oracle.py')
    cases = [json.loads(line) for line in (args.leaf/'timing.jsonl').read_text().splitlines() if line.strip()]
    records = []
    for name in args.case:
        case = next(row for row in cases if row['name'] == name)
        inputs = oracle.gen_inputs(case, 'npu:0')
        candidate.run(**inputs)
        torch.npu.synchronize()
        profiler = cProfile.Profile()
        wall = []
        for _ in range(args.iterations):
            # 输入构造、同步和计时器均在采样区间外；每次使用新输入。
            inputs = oracle.gen_inputs(case, 'npu:0')
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            profiler.enable()
            candidate.run(**inputs)
            profiler.disable()
            wall.append(time.perf_counter_ns()-start)
            torch.npu.synchronize()
        rows = []
        for (file, line, function), (primitive, total, own, cumulative, _) in pstats.Stats(profiler).stats.items():
            rows.append(dict(file=file, line=line, function=function, primitive_calls=primitive,
                total_calls=total, self_seconds=own, cumulative_seconds=cumulative))
        rows.sort(key=lambda row: row['cumulative_seconds'], reverse=True)
        records.append(dict(case=name, iterations=args.iterations, profiled_host_wall_ns=wall,
            top_cumulative=rows[:50], top_self=sorted(rows, key=lambda row: row['self_seconds'], reverse=True)[:30]))
    launcher = getattr(driver, 'AscendLauncher', None)
    source = Path(inspect.getfile(driver))
    result = dict(source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), records=records,
        backend=dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            launcher_class=inspect.getsource(launcher) if launcher is not None else None),
        scope='采样候选热调用的宿主函数栈；cProfile有额外开销，不作为正式加速比，也不把累计时间相加当作总时长')
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(status='completed', cases=len(records))))


if __name__ == '__main__':
    main()
