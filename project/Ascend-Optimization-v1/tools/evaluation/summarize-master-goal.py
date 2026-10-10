#!/usr/bin/env python3
"""重新核验三个算子的四版本原始证据，生成中文总交付表。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

MASTER = 'd6a8eec473517a3d68157b208eb9c057eb1d4c50'
OPERATORS = {
    'amin': ('G92', 0.5536, 27, 12, 15, 'master-compatible',
             'master-grid-adapter-20261010/flaggems-master-grid-adapted-1.result.json',
             'grid-master-20261010'),
    'matmul_bias_activation': ('G611', 0.5408, 42, 27, 15, 'master',
                             'master-20261010/flaggems-master-1.result.json', 'master-20261010'),
    'narrow_copy': ('G29', 0.0138, 33, 18, 15, 'master',
                    'master-case-adapter-20261010/flaggems-master-1.result.json', 'master-20261010'),
}


def result_path(project, operator, recorded):
    # 比较文件可能由 Windows 或 Linux 生成；仅接受该算子 reports 内的路径。
    marker = 'operators/' + operator + '/reports/'
    normalized = str(recorded).replace('\\', '/')
    if normalized.count(marker) != 1:
        raise ValueError('结果路径不属于指定算子 reports：' + normalized)
    root = (project / 'operators' / operator / 'reports').resolve()
    path = (root / normalized.split(marker, 1)[1]).resolve()
    path.relative_to(root)
    return path


def audit(project):
    tool = Path(__file__).with_name('compare-operator-versions.py')
    spec = importlib.util.spec_from_file_location('version_comparison', tool)
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    operators = []
    for operator, (cell, excel, count, correctness, timing, baseline_name, baseline_file, suffix) in OPERATORS.items():
        reports = project / 'operators' / operator / 'reports'
        closure = json.loads((reports / 'master-closure-20261010/版本对比.json').read_text(encoding='utf-8'))
        expected_names = [baseline_name, 'kg-no-profile', 'kg-profile', 'optimized']
        if [r.get('version') for r in closure] != expected_names:
            raise ValueError(operator + ' 缺少完整四版本或顺序错误')
        paths = [result_path(project, operator, row['result_path']) for row in closure]
        expected = [reports / baseline_file]
        for arm in ('no-profile', 'profile'):
            expected.append(reports / ('kg-' + arm + '-' + suffix) / 'independent' / ('kg-' + arm + '-1.result.json'))
        if paths[:3] != [p.resolve() for p in expected]:
            raise ValueError(operator + ' 基线或原生组未使用固定独立复验结果')
        versions = [(name, comparison.load_result(path)) for name, path in zip(expected_names, paths)]
        candidate_name = ('flaggems-master-grid-adapter-20261010' if operator == 'amin'
                          else 'flaggems-master-20261010')
        candidate = project / 'operators' / operator / 'candidates' / (candidate_name + '.py')
        origin = json.loads(candidate.with_suffix('.provenance.json').read_text(encoding='utf-8'))
        if (origin.get('upstream_commit') != MASTER
                or origin.get('candidate_sha256') != versions[0][1]['source_sha256']
                or hashlib.sha256(candidate.read_bytes()).hexdigest() != versions[0][1]['source_sha256']):
            raise ValueError(operator + ' 固定 master 候选来源或 SHA 不匹配')
        recomputed = comparison.compare(versions)
        for stored, row in zip(closure, recomputed):
            if row['correctness_count'] != correctness or row['timing_count'] != timing:
                raise ValueError(operator + ' 完整用例数与合同不一致')
            for key in ('source_sha256', 'result_sha256', 'relative_pytorch', 'relative_master'):
                if stored.get(key) != row[key]:
                    raise ValueError(operator + ' 比较摘要与原始证据不一致：' + key)
            row['result_path'] = Path(row['result_path']).resolve().relative_to(project.resolve()).as_posix()
        operators.append({'operator': operator, 'excel_cell': cell, 'excel_speedup': excel,
                          'total_cases': count, 'versions': recomputed})
    return {'upstream_commit': MASTER, 'timing_scope': 'FlagGems core device_kernel',
            'aggregation': '逐性能 case 加速比的几何平均', 'operators': operators}


def render(data):
    lines = ['# 低于 0.8× 算子四版本闭环', '',
             '全部数值由同一固定合同的完整独立评测重算。性能口径为设备 kernel 计时，逐 case 加速比取几何平均。', '',
             '| 算子 | Excel 历史记录 | 固定 master | KG 无 profiler | KG 原生 profiler | 我们的优化版 | 相对 master | 正确性与性能 |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for op in data['operators']:
        rows = op['versions']
        values = ' | '.join(f"{r['relative_pytorch']:.3f}×" for r in rows)
        optimized = rows[-1]
        lines.append(f"| {op['operator']} | {op['excel_cell']}：{op['excel_speedup']:.4f}× | {values} | "
                     f"{optimized['relative_master']:.3f}× | {op['total_cases']}/{op['total_cases']} 全通过 |")
    lines += ['', 'amin 的固定 master 列指保留原内核与 autotune 的启动兼容修复基线；未修改版本的大输入启动超限失败记录保留。',
              'narrow_copy 只适配 benchmark case API，算子代码仍来自固定 master。Excel 历史数据不与本轮合同数据混算。', '',
              '## 复现入口', '', f"FlagGems commit：`{data['upstream_commit']}`。", '']
    for op in data['operators']:
        lines += [f"### {op['operator']}", '']
        for r in op['versions']:
            lines.append(f"- {r['version']}：`{r['result_path']}`；源码 SHA `{r['source_sha256']}`；"
                         f"PyTorch 参考漂移 {r['reference_drift_geomean']:.3f}×。")
        lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    # 任何一个算子缺失都会在写入输出之前失败。
    data = audit(args.project)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data), encoding='utf-8')
    args.output.with_suffix('.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('THREE_OPERATOR_FOUR_VERSION_AUDIT_PASSED')


if __name__ == '__main__':
    main()
