#!/usr/bin/env python3
"""核对固定 FlagGems 源码及 KGS 声明的 FlagTree/运行时身份。"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = Path(__file__).resolve().parents[5]


def verify(repo, status=None):
    lock = json.loads((repo / 'manifests/hgkda-stage2.lock.json').read_text(encoding='utf-8'))
    manifest = json.loads((repo / 'manifests/flaggems-master-files.json').read_text(encoding='utf-8'))
    assert manifest['upstream_commit'] == lock['flaggems']['commit'], 'FlagGems 提交不匹配'
    bad = []
    for row in manifest['files']:
        path = repo / 'third_party/FlagGems' / row['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            bad.append(row['path'])
    if bad:
        raise ValueError('固定 FlagGems 文件不匹配：' + ', '.join(bad[:10]))
    if status is not None:
        actual = status.get('software', {})
        for field, expected in lock['service_software'].items():
            if actual.get(field) != expected:
                raise ValueError(f'运行时版本不匹配：{field}={actual.get(field)!r}，要求 {expected!r}')
        if status.get('target', {}).get('backend') != 'ascend':
            raise ValueError('KGS 不是锁定的 Ascend 后端')
    return {'status': 'verified', 'flaggems_commit': manifest['upstream_commit'],
            'flaggems_files_checked': len(manifest['files']),
            'service_software_checked': status is not None,
            'service_software': status.get('software') if status else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=REPO)
    p.add_argument('--server')
    p.add_argument('--service-status', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.server and a.service_status:
        p.error('--server 与 --service-status 二选一')
    status = None
    if a.server:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        status = json.load(opener.open(a.server.rstrip('/') + '/status', timeout=20))
    elif a.service_status:
        status = json.loads(a.service_status.read_text(encoding='utf-8-sig'))
    data = verify(a.repo, status)
    if a.output:
        if a.output.exists():
            p.error('输出已存在，请保留旧核验记录并使用新路径')
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(data, ensure_ascii=False))


if __name__ == '__main__':
    main()
