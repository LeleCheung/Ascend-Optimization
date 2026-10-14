#!/usr/bin/env python3
"""同卡串行复测 Excel 低于0.8的候选，只将正确且仍慢的算子立项。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT.parents[1]
CANDIDATES = [('logaddexp2', 'G122', 0.3218, 'logaddexp2.py', 'logaddexp2'),
              ('acosh', 'G88', 0.6589, 'acosh.py', 'acosh'),
              ('add_relu', 'G17', 0.5174, '_add_relu.py', '_add_relu')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', default='20261012-agent')
    args = parser.parse_args()
    records = []
    for name, cell, excel, filename, entry in CANDIDATES:
        operator = PROJECT / 'operators' / name
        candidates = operator / 'candidates'
        candidates.mkdir(parents=True, exist_ok=True)
        upstream = REPO / 'third_party/FlagGems/src/flag_gems/ops' / filename
        source = upstream.read_text(encoding='utf-8') + '\n\nrun = ' + entry + '\n'
        target = candidates / ('flaggems-master-' + args.batch + '.py')
        if target.exists() and target.read_text(encoding='utf-8') != source:
            raise ValueError('基线已存在不同源码')
        target.write_bytes(source.encode())
        report = operator / 'reports' / ('screen-master-' + args.batch)
        command = [sys.executable, str(PROJECT / 'tools/evaluation/evaluate-operator.py'),
                   '--operator', name, '--source', str(target), '--output', str(report),
                   '--label', 'master', '--timeout', '1800', '--timing-scope', 'device_kernel']
        result = subprocess.run(command)
        raw = report / 'master-1.result.json'
        if not raw.exists():
            raise RuntimeError('评测缺少最终结果，停止同卡筛选')
        evaluated = json.loads(raw.read_text(encoding='utf-8'))
        record = {'operator': name, 'excel_cell': cell, 'excel_speedup': excel,
                  'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                  'status': evaluated.get('status'), 'geo_mean': evaluated.get('geo_mean'),
                  'result': str(raw), 'source': str(target)}
        records.append(record)
        output = PROJECT / 'operators' / ('next-operator-screen-' + args.batch + '.json')
        output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
        if (result.returncode == 0 and evaluated['status'] == 'PASSED'
                and not evaluated.get('is_hack') and evaluated['num_passed'] == evaluated['num_workloads']
                and evaluated['geo_mean'] < 0.8):
            record['selected'] = True
            output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
            (operator / 'README.md').write_text(
                f'# {name}\n\n验收看板 {cell} 为 {excel}×。本轮固定 master 复测 {evaluated["geo_mean"]:.3f}×，'
                f'{evaluated["num_passed"]}/{evaluated["num_workloads"]} 全通过，作为 Profiling Agent 的新正式对象。\n\n'
                f'基线源码：`candidates/{target.name}`。评测证据：`reports/{report.name}/master-1.result.json`。\n'
                '接下来运行原生 KG 无 profiler/有 profiler两组及独立 Agent 闭环，保持完整合同。\n', encoding='utf-8')
            print('SELECTED', name, flush=True)
            return
        if evaluated.get('status') in ('HTTP_ERROR', 'TIMEOUT', 'RUNTIME_ERROR'):
            raise RuntimeError('设备或接口失败，停止筛选并保留证据')
    print('NO_ELIGIBLE_OPERATOR_IN_SHORTLIST', flush=True)


if __name__ == '__main__':
    main()
