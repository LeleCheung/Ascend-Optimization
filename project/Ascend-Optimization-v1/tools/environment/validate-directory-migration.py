#!/usr/bin/env python3
"""核验迁移后的三项完整评测与旧批次的源码、用例和计时口径一致。"""
import argparse
import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def validate(project):
    comparison = module('comparison', project / 'tools/evaluation/compare-operator-versions.py')
    summary = module('summary', project / 'tools/evaluation/summarize-master-goal.py')
    rows = []
    for operator, count in [('amin', 27), ('matmul_bias_activation', 42), ('narrow_copy', 33)]:
        reports = project / 'operators' / operator / 'reports'
        closure = json.loads((reports / 'master-closure-20261010/版本对比.json').read_text(encoding='utf-8'))
        recorded = next(r for r in closure if r['version'] == 'optimized')
        old_path = summary.result_path(project, operator, recorded['result_path'])
        path = reports / 'directory-migration-20261012/optimized-1.result.json'
        old = comparison.load_result(old_path)
        current = comparison.load_result(path)
        if current['source_sha256'] != old['source_sha256'] or len(current['rows']) != count:
            raise ValueError(operator + ' 源码或完整范围改变')
        for key in ('binding', 'device', 'timing_scope'):
            if current[key] != old[key]:
                raise ValueError(operator + ' 评测口径改变：' + key)
        if set(current['rows']) != set(old['rows']):
            raise ValueError(operator + ' 完整用例集合改变')
        for case, row in current['rows'].items():
            if any(row.get(k) != old['rows'][case].get(k) for k in ('axes', 'phase')):
                raise ValueError(operator + ' 用例参数改变：' + case)
        old_inspect = json.loads((old_path.parent / 'inspect.json').read_text(encoding='utf-8'))
        current_inspect = json.loads((path.parent / 'inspect.json').read_text(encoding='utf-8'))
        if (current_inspect['case_list']['cases'] != old_inspect['case_list']['cases']
                or current_inspect['candidate_contract'] != old_inspect['candidate_contract']):
            raise ValueError(operator + ' 性能合同改变')
        speedup = statistics.geometric_mean(
            r['reference_latency_ms'] / r['latency_ms'] for r in current['timing'].values())
        result = json.loads(path.read_text(encoding='utf-8'))
        reported_speedup = statistics.geometric_mean(r['speedup'] for r in current['timing'].values())
        if not math.isclose(reported_speedup, result['geo_mean'], rel_tol=1e-10):
            raise ValueError(operator + ' 汇总加速比与逐 case 数据矛盾')
        rows.append({'operator': operator, 'status': 'PASSED', 'count': count,
                     'previous_speedup': recorded['relative_pytorch'], 'speedup': speedup,
                     'server_geo_mean': result['geo_mean'],
                     'source_sha256': current['source_sha256'],
                     'result_path': path.relative_to(project).as_posix(),
                     'result_sha256': current['result_sha256'],
                     'previous_benchmark_fingerprint': old['fingerprint'],
                     'benchmark_fingerprint': current['fingerprint']})
    return {'operators': rows, 'fingerprint_note':
            'KGS 指纹包含评测树 Git HEAD；由独立 checkout 改为总仓库/生成快照后发生变化。'
            '源码、完整 case 参数和计时口径已分别核验；历史原始指纹不改写。'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.project.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
