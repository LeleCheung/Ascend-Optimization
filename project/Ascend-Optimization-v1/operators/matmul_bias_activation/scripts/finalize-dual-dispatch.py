#!/usr/bin/env python3
"""完成 v8 两轮复验后，与 v6 比较并重建正式四版本报告。"""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    tools = project / 'tools/evaluation'
    spec = importlib.util.spec_from_file_location('comparison', tools / 'compare-operator-versions.py')
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    reports = project / 'operators/matmul_bias_activation/reports'
    baseline = reports / 'master-20261010/flaggems-master-1.result.json'
    base = comparison.load_result(baseline)
    v6 = reports / 'profiling-kgroup-v6-20261010/profiling-kgroup-v6-1.result.json'
    v8 = sorted((reports / 'profiling-dual-dispatch-v8-20261010').glob('*.result.json'))
    assert len(v8) == 2, 'v8 两轮完整结果尚未齐全'
    choices = []
    for label, results in [('v6', [v6]), ('v8', v8)]:
        valid = []
        for path in results:
            if json.loads(path.read_text(encoding='utf-8')).get('status') != 'PASSED':
                break
            loaded = comparison.load_result(path)
            score = comparison.compare([('master', base), ('optimized', loaded)])[-1]
            assert score['correctness_count'] == 27 and score['timing_count'] == 15
            valid.append((score['relative_pytorch'], path, loaded['source_sha256']))
        else:
            assert len({row[2] for row in valid}) == 1
            representative = min(valid, key=lambda row: row[0])
            choices.append((representative[0], representative[1], label, len(valid)))
    score, best, label, repeats = max(choices, key=lambda row: row[0])
    command = [sys.executable, str(tools / 'compare-operator-versions.py'),
               '--version', 'master', str(baseline)]
    for arm in ('no-profile', 'profile'):
        command += ['--version', 'kg-' + arm,
                    str(reports / ('kg-' + arm + '-master-20261010') /
                        'independent' / ('kg-' + arm + '-1.result.json'))]
    command += ['--version', 'optimized', str(best),
                '--output', str(reports / 'master-closure-20261010/版本对比.md')]
    subprocess.run(command, check=True)
    selected = {'version': label, 'relative_pytorch': score, 'repeats': repeats,
                'representative_result': str(best.relative_to(project)),
                'selection': '同源码重复测试取较慢一轮；候选之间比较完整独立复验',
                'candidates': [{'version': row[2], 'relative_pytorch': row[0],
                                'result': str(row[1].relative_to(project)), 'repeats': row[3]}
                               for row in choices]}
    (reports / 'master-closure-20261010/最终候选.json').write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('MATMUL_FINAL_CANDIDATE', json.dumps(selected, ensure_ascii=False))


if __name__ == '__main__':
    main()
