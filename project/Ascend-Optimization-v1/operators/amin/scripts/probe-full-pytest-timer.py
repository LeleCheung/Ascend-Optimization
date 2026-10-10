#!/usr/bin/env python3
"""经 KGS debug 队列复现完整 amin pytest，记录每个计时调用和原始报告。

将本脚本与固定 master 的入口适配候选一起传给 debug-operator.py。
这是诊断，不能替代正式 /evaluate 的完整结果。
"""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path('/data/hanle/ascend-optimization/goal-20261010/FlagGems')
    candidate = Path('flaggems-master-dim-adapter-v2-20261010.py').resolve()
    artifacts = Path(os.environ['KGS_DEBUG_ARTIFACTS'])
    plugin = candidate.parent / 'amin_full_timer_plugin.py'
    plugin.write_text('''
def pytest_collection_modifyitems(session, config, items):
    import benchmark.base as b
    import torch
    import os
    from pathlib import Path
    import triton.backends.ascend.testing as testing
    print('CONFIG', b.Config.mode, b.Config.warm_up, b.Config.repetition, flush=True)
    # 只保留同一计时器的原始数据，不修改预热、迭代或延迟计算。
    timer = testing.do_bench_npu
    timer_index = 0
    def retained_timer(*args, **kwargs):
        nonlocal timer_index
        timer_index += 1
        directory = Path(os.environ['KGS_DEBUG_ARTIFACTS']) / ('timer-%03d' % timer_index)
        kwargs.update(prof_dir=str(directory), keep_res=True)
        print('TIMER_ARTIFACTS', str(directory), flush=True)
        return timer(*args, **kwargs)
    testing.do_bench_npu = retained_timer
    original = b.Benchmark.get_latency
    def latency(self, op, *args, **kwargs):
        print('OP_START', str(op), [(tuple(x.shape), str(x.dtype)) for x in args if isinstance(x, torch.Tensor)], flush=True)
        try:
            result = original(self, op, *args, **kwargs)
            print('OP_LATENCY', result, flush=True)
            return result
        except BaseException as error:
            print('OP_FAILED', str(op), type(error).__name__, str(error), flush=True)
            raise
    b.Benchmark.get_latency = latency
''', encoding='utf-8')
    env = dict(os.environ)
    env['GEMS_VENDOR'] = 'ascend'
    env['PYTHONPATH'] = os.pathsep.join([
        str(candidate.parent), str(root / 'src'), str(root), env.get('PYTHONPATH', '')])
    bootstrap = ("import runpy,site,sys;"
                 "sys.path.extend(p for p in site.getsitepackages() if p not in sys.path);"
                 "runpy.run_module('pytest',run_name='__main__',alter_sys=True)")
    stages = [
        ('accuracy', ['tests/test_amin.py']),
        ('benchmark', ['benchmark/test_amin.py', '--level', 'core',
                       '--warmup', '1000', '--iter', '100', '-p', 'amin_full_timer_plugin']),
    ]
    summaries = []
    for name, args in stages:
        report = artifacts / (name + '.json')
        command = [sys.executable, '-s', '-c', bootstrap, '-q', '-s', *args,
                   '-m', 'amin', '--record', 'json', '--output', str(report),
                   '--override', 'amin:' + str(candidate) + ':run']
        print('STAGE_START', name, flush=True)
        result = subprocess.run(command, cwd=root, env=env, capture_output=True,
                                text=True, timeout=900)
        (artifacts / (name + '.stdout.txt')).write_text(result.stdout, encoding='utf-8')
        (artifacts / (name + '.stderr.txt')).write_text(result.stderr, encoding='utf-8')
        for line in result.stdout.splitlines():
            if line.startswith(('CONFIG', 'OP_', 'TIMER_', 'SUCCESS', 'FAILED')) or 'flow events' in line:
                print(line, flush=True)
        print('STAGE_EXIT', name, result.returncode, flush=True)
        summaries.append({'stage': name, 'exit_code': result.returncode,
                          'report_exists': report.exists()})
    (artifacts / 'stages.json').write_text(json.dumps(summaries, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
