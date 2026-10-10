#!/usr/bin/env python3
"""只读查看独立 campaign 的 KG 状态、完整结果和指定 KGS 队列。"""
import argparse
import datetime
import json
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--server', action='append', default=[])
    args = parser.parse_args()
    root = args.root.resolve()
    data = {'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'root': str(root), 'kg': [], 'evaluations': [], 'servers': []}
    for path in sorted((root / 'kg-runs').glob('*/.kernelgen/run-progress.json')):
        progress = json.loads(path.read_text(encoding='utf-8'))
        workspace = path.parent.parent
        row = {'workspace': workspace.name,
               **{key: progress.get(key) for key in ('state', 'stage', 'updated_at', 'stop_reason')}}
        active = path.with_name('active-server-operations.json')
        if active.exists():
            # KG 可能在终态仍留下旧 operation 条目；是否正在执行以 KGS 队列为准。
            row['recorded_operations'] = json.loads(active.read_text(encoding='utf-8')).get('operations', {})
        rounds = []
        for result in sorted(workspace.glob('stages/optimize/work/.kernelgen/evals/round-*/result.json')):
            record = json.loads(result.read_text(encoding='utf-8'))
            rounds.append({'round': result.parent.name,
                           **{key: record.get(key) for key in ('status', 'geo_mean', 'num_passed', 'num_workloads')}})
        row['rounds'] = rounds
        data['kg'].append(row)
    project = root / 'Ascend-Optimization/project/Ascend-Optimization-v1'
    for operator in ('amin', 'matmul_bias_activation', 'narrow_copy'):
        reports = project / 'operators' / operator / 'reports'
        for directory in sorted(reports.iterdir()):
            if not directory.is_dir() or '20261010' not in directory.name:
                continue
            for result in sorted(directory.glob('**/*.result.json')):
                record = json.loads(result.read_text(encoding='utf-8'))
                data['evaluations'].append({'operator': operator,
                    'path': result.relative_to(project).as_posix(),
                    **{key: record.get(key) for key in ('status', 'geo_mean', 'num_passed', 'num_workloads')}})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for server in args.server:
        try:
            with opener.open(server.rstrip('/') + '/status', timeout=10) as response:
                record = json.load(response)
            data['servers'].append({'server': server, 'status': record.get('status'),
                                     'scheduler': record.get('scheduler')})
        except Exception as error:
            data['servers'].append({'server': server, 'error': str(error)})
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
