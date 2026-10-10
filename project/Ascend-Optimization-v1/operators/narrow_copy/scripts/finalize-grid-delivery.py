#!/usr/bin/env python3
"""从完整复验中保守择优，重新生成 narrow 四版本及三算子总表。"""
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
    reports = project / 'operators/narrow_copy/reports'
    baseline = reports / 'master-grid-device-20261010/flaggems-master-grid-1.result.json'
    base = comparison.load_result(baseline)
    choices = []
    for directory, kind in (
        ('contiguous-copy-grid-device-20261010', '纯 Triton'),
        ('small-dma-grid-device-20261010', '小输入 CANN DMA + 大输入 Triton'),
        ('contiguous-dma-grid-device-20261010', '所有连续 dim=0 片段使用 CANN DMA，其余使用 Triton'),
        ('profile-small-dma-grid-device-20261010', '小输入 CANN DMA + KG profiler 循环复制 Triton'),
    ):
        results = sorted((reports / directory).glob('*.result.json'))
        if not results:
            continue
        valid = []
        for path in results:
            raw = json.loads(path.read_text(encoding='utf-8'))
            if raw.get('status') != 'PASSED':
                break
            loaded = comparison.load_result(path)
            score = comparison.compare([('master-compatible', base), ('optimized', loaded)])[-1]
            assert score['correctness_count'] == 18 and score['timing_count'] == 15
            valid.append((score['relative_pytorch'], path, loaded['source_sha256']))
        else:
            assert len({row[2] for row in valid}) == 1, '同一候选的重复评测源码不同'
            # 重复评测采用较慢一轮作为代表，不把偶然最高值写为最终成绩。
            representative = min(valid, key=lambda row: row[0])
            choices.append((representative[0], representative[1], kind, len(valid)))
    assert choices, '没有完整通过的分析候选'
    score, best, kind, repeats = max(choices, key=lambda row: row[0])
    command = [sys.executable, str(tools / 'compare-operator-versions.py'),
               '--version', 'master-compatible', str(baseline)]
    for arm in ('no-profile', 'profile'):
        command += ['--version', 'kg-' + arm,
                    str(reports / ('kg-' + arm + '-grid-device-master-20261010') /
                        'independent' / ('kg-' + arm + '-1.result.json'))]
    command += ['--version', 'optimized', str(best),
                '--output', str(reports / 'master-closure-20261010/版本对比.md')]
    subprocess.run(command, check=True)
    selected = {'implementation_kind': kind, 'representative_result': str(best.relative_to(project)),
                'selection': '候选之间比较完整独立复验；同源码重复评测以较慢一轮为代表',
                'repeats': repeats, 'relative_pytorch': score,
                'candidates': [{'score': row[0], 'result': str(row[1].relative_to(project)),
                                'implementation_kind': row[2], 'repeats': row[3]} for row in choices]}
    (reports / 'master-closure-20261010/最终候选.json').write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    subprocess.run([sys.executable, str(tools / 'summarize-master-goal.py'), str(project),
                    str(project / 'operators/低于0.8算子四版本闭环-20261010.md')], check=True)
    print('NARROW_FINAL_CANDIDATE', json.dumps(selected, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
