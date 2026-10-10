#!/usr/bin/env python3
"""离线核验本轮原始附件、KG 归档与完整评测源码，不使用 NPU。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def verify_file(base, name, metadata):
    path = (base / name).resolve()
    path.relative_to(base.resolve())
    data = path.read_bytes()
    assert len(data) == metadata['size_bytes'], path
    assert hashlib.sha256(data).hexdigest() == metadata['sha256'], path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    spec = importlib.util.spec_from_file_location(
        'comparison', Path(__file__).with_name('compare-operator-versions.py'))
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    checked = 0
    for operator in ('amin', 'matmul_bias_activation', 'narrow_copy'):
        root = project / 'operators' / operator / 'reports'
        assert root.is_dir(), root
        for manifest in root.glob('*20261010/**/archive-manifest.json'):
            data = json.loads(manifest.read_text(encoding='utf-8'))
            for row in data['files']:
                verify_file(manifest.parent, row['archived_path'], row)
                checked += 1
        for response in root.glob('*20261010/**/response.json'):
            data = json.loads(response.read_text(encoding='utf-8'))
            base = response.parent / 'artifacts'
            if data.get('job_id'):
                rows = data.get('artifacts', [])
            else:
                manifest = base / 'manifest.json'
                rows = json.loads(manifest.read_text(encoding='utf-8')).get(
                    'artifacts', []) if manifest.exists() else []
            for row in rows:
                name = row['path'] if data.get('job_id') else Path(row['path']).name
                verify_file(base, name, row)
                checked += 1
        for result in root.glob('*20261010/**/*.result.json'):
            data = json.loads(result.read_text(encoding='utf-8'))
            if data.get('status') == 'PASSED':
                comparison.load_result(result)
                checked += 1
    assert checked > 0, '未找到本轮证据'
    print('本轮证据与完整评测核验通过', checked)


if __name__ == '__main__':
    main()
