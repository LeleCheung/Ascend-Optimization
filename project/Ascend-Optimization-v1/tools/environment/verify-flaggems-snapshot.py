#!/usr/bin/env python3
"""核验固定 FlagGems 的磁盘字节和 Git 跟踪清单，包含上游已跟踪的忽略文件。"""
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def main():
    manifest = json.loads((REPO / 'manifests/flaggems-master-files.json').read_text(encoding='utf-8'))
    tracked = set(subprocess.check_output([
        'git', '-C', str(REPO), 'ls-files', '-z', '--', 'third_party/FlagGems'
    ]).decode().strip('\0').split('\0'))
    expected = {'third_party/FlagGems/' + row['path'] for row in manifest['files']}
    if tracked != expected:
        raise ValueError('Git 跟踪清单不一致：缺失 ' + repr(sorted(expected - tracked))
                         + '；多余 ' + repr(sorted(tracked - expected)))
    for row in manifest['files']:
        path = REPO / 'third_party/FlagGems' / row['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('源码字节不一致：' + row['path'])
    print(json.dumps({'upstream_commit': manifest['upstream_commit'], 'verified_files': len(expected)}))


if __name__ == '__main__':
    main()
