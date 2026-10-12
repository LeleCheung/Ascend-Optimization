#!/usr/bin/env python3
"""从已核验的远端 Git checkout 导入固定 FlagGems 源码，不带嵌套 Git。"""
import argparse
import hashlib
import importlib.util
import io
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

MASTER = 'd6a8eec473517a3d68157b208eb9c057eb1d4c50'
REPO = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('control', Path(__file__).parents[1] / 'control/control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--checkout', required=True)
    args = parser.parse_args()
    data = control.remote_python(args.host, {'checkout': args.checkout, 'commit': MASTER}, r'''
import subprocess,sys
checkout=parameters['checkout']; commit=parameters['commit']
assert subprocess.check_output(['git','-C',checkout,'rev-parse','HEAD']).decode().strip()==commit
assert not subprocess.check_output(['git','-C',checkout,'diff','HEAD','--'])
sys.stdout.buffer.write(subprocess.check_output(['git','-C',checkout,'archive','--format=tar.gz',commit]))
''')
    state = REPO / '.control-state/flaggems-import-20261012'
    state.mkdir(parents=True, exist_ok=False)
    (state / 'master.tar.gz').write_bytes(data)
    stage = state / 'new'
    stage.mkdir()
    rows = []
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        members = archive.getmembers()
        seen = set()
        for member in members:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or '..' in name.parts or '.git' in name.parts
                    or '\\' in member.name or ':' in member.name
                    or not (member.isfile() or member.isdir()) or member.name in seen):
                raise ValueError('不安全的上游归档条目：' + member.name)
            seen.add(member.name)
        for member in members:
            target = stage.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            content = archive.extractfile(member).read()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            rows.append({'path': member.name, 'sha256': hashlib.sha256(content).hexdigest(),
                         'size': len(content), 'mode': member.mode})
    manifest = {'upstream': 'https://github.com/flagos-ai/FlagGems',
                'upstream_commit': MASTER, 'files': rows}
    (REPO / 'manifests/flaggems-master-files.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    current = REPO / 'third_party/FlagGems'
    shutil.move(str(current), str(state / 'historical'))
    shutil.move(str(stage), str(current))
    print(json.dumps({'commit': MASTER, 'files': len(rows), 'historical_backup': str(state / 'historical')}))


if __name__ == '__main__':
    main()
