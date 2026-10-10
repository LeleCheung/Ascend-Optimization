#!/usr/bin/env python3
"""核验同一 KGS 合同的完整结果，生成逐 case CSV 与中文版本对比报告。"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_result(path):
    result = read(path)
    if result.get('status') != 'PASSED' or result.get('is_hack'):
        raise ValueError(f'结果未完整通过：{path}')
    rows = result.get('per_workload', [])
    ids = {row['uuid'] for row in rows}
    if (len(ids) != len(rows) or not rows or len(rows) != result.get('num_workloads')
            or result.get('num_passed') != len(rows)
            or any(row.get('status') != 'PASSED' for row in rows)):
        raise ValueError(f'完整通过与逐 case 记录矛盾：{path}')
    if not any(row.get('phase') == 'correctness' for row in rows):
        raise ValueError(f'缺少完整正确性结果：{path}')
    timing = {row['uuid']: row for row in rows if row.get('phase') == 'timing'}
    for row in timing.values():
        for key in ('latency_ms', 'reference_latency_ms'):
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'无效延迟：{path} {row["uuid"]} {key}')
    if not timing:
        raise ValueError(f'缺少性能结果：{path}')
    label = path.name.removesuffix('.result.json').rsplit('-', 1)[0]
    request = read(path.with_name(path.name.replace('.result.json', '.request.json')))
    provenance = read(path.with_name(label + '.provenance.json'))
    sources = request.get('implementation', {}).get('sources', [])
    if len(sources) != 1 or sources[0].get('path') != 'main.py':
        raise ValueError('只支持单文件 main.py 候选')
    digest = hashlib.sha256(sources[0]['content'].encode()).hexdigest()
    if digest != provenance.get('source_sha256') or request.get('binding') != provenance.get('binding'):
        raise ValueError(f'源码或合同与 provenance 不一致：{path}')
    if sources[0]['content'] != path.with_name(label + '.py').read_text(encoding='utf-8'):
        raise ValueError(f'候选快照与评测请求不一致：{path}')
    inspect = read(path.parent / 'inspect.json')
    expected = {case['case_id'] for case in inspect['case_list']['cases']}
    if set(timing) != expected:
        raise ValueError(f'性能 case 未完整覆盖 inspect：{path}')
    return {'path': str(path), 'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_sha256': digest, 'binding': request['binding'],
            'fingerprint': inspect['benchmark_fingerprint'], 'device': result.get('device'),
            'timing': timing, 'rows': {row['uuid']: row for row in rows},
            'correctness_count': len(rows) - len(timing)}


def compare(versions):
    master = versions[0][1]
    output = []
    for name, current in versions:
        for key in ('binding', 'fingerprint', 'device'):
            if current[key] != master[key]:
                raise ValueError(f'{name} 与 master 的 {key} 不同')
        if set(current['rows']) != set(master['rows']):
            raise ValueError(f'{name} 与 master 的完整用例不同')
        for case_id, row in current['rows'].items():
            base = master['rows'][case_id]
            if row.get('axes') != base.get('axes') or row.get('phase') != base.get('phase'):
                raise ValueError(f'{name} 的 case 参数不同：{case_id}')
        timing = current['timing']
        torch_ratios = [r['reference_latency_ms'] / r['latency_ms'] for r in timing.values()]
        master_ratios = [master['timing'][case]['latency_ms'] / r['latency_ms']
                         for case, r in timing.items()]
        drift = [r['reference_latency_ms'] / master['timing'][case]['reference_latency_ms']
                 for case, r in timing.items()]
        output.append({'version': name, 'relative_pytorch': statistics.geometric_mean(torch_ratios),
                       'relative_master': statistics.geometric_mean(master_ratios),
                       'reference_drift_geomean': statistics.geometric_mean(drift),
                       'reference_drift_min': min(drift), 'reference_drift_max': max(drift),
                       'correctness_count': current['correctness_count'], 'timing_count': len(timing),
                       'source_sha256': current['source_sha256'], 'result_path': current['path'],
                       'result_sha256': current['result_sha256']})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='append', nargs=2, required=True,
                        metavar=('NAME', 'RESULT'), help='第一项为固定 master；仅接收独立完整复验结果')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len({name for name, _ in args.version}) != len(args.version):
        parser.error('版本名称不能重复')
    versions = [(name, load_result(Path(path))) for name, path in args.version]
    summary = compare(versions)
    lines = ['# 同合同版本性能对比', '',
             '以下为已提供并通过独立完整评测的版本。计时为 FlagGems core 设备 kernel 计时；数值为逐 case 加速比的几何平均。', '',
             '| 版本 | 正确性 | 性能 case | 相对 PyTorch | 相对固定 master | PyTorch 基线变化 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in summary:
        lines.append(f'| {row["version"]} | {row["correctness_count"]} 项全通过 | {row["timing_count"]} | '
                     f'{row["relative_pytorch"]:.3f}× | {row["relative_master"]:.3f}× | '
                     f'{row["reference_drift_geomean"]:.3f}× |')
    lines += ['', '相对 master 使用相同 case 的 master 候选延迟 ÷ 当前候选延迟。PyTorch 基线变化用于检查各批次的环境波动；逐 case 数据见同名 CSV。', '']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines), encoding='utf-8')
    args.output.with_suffix('.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with args.output.with_suffix('.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['版本', 'case_id', 'dtype', 'shape_detail', 'PyTorch_us', '候选_us', '相对PyTorch', '相对master', 'PyTorch基线变化'])
        master = versions[0][1]
        for name, version in versions:
            for case_id, row in version['timing'].items():
                base = master['timing'][case_id]
                writer.writerow([name, case_id, row['axes'].get('dtype'), json.dumps(row['axes'].get('shape_detail')),
                                 row['reference_latency_ms'] * 1000, row['latency_ms'] * 1000,
                                 row['reference_latency_ms'] / row['latency_ms'],
                                 base['latency_ms'] / row['latency_ms'],
                                 row['reference_latency_ms'] / base['reference_latency_ms']])
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
