"""Wait for frozen replays, verify exports, rebuild and check presentation artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time


RUNS = ['presentation-replay-20261119', 'presentation-scatter-replay-20261119',
        'presentation-scatter-native-adapter-20261119']


def remote(host, source):
    process = subprocess.run(['ssh', host, 'python3', '-'], input=source.encode(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return json.loads(process.stdout)


def unpack_verified(archive_path, reports):
    with tarfile.open(archive_path) as archive:
        members = {item.name: item for item in archive.getmembers()}
        receipt_name = next(name for name in members if name.endswith('/export-receipt.json'))
        prefix = receipt_name.rsplit('/', 1)[0]
        receipt = json.loads(archive.extractfile(members[receipt_name]).read())
        for item in receipt['files']:
            name = prefix + '/' + item['path']
            raw = archive.extractfile(members[name]).read()
            assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
        for name, member in members.items():
            target = (reports / name).resolve()
            assert target.is_relative_to(reports.resolve()) and member.isfile()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
    return len(receipt['files'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='baai-910b')
    parser.add_argument('--remote-base', default='/data/hanle/ascend-optimization/runs')
    parser.add_argument('--remote-python', default='/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python')
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    previous = None
    while True:
        code = ('import json\nfrom pathlib import Path\nbase=Path(' + repr(args.remote_base) + ')\n'
            'runs=' + repr(RUNS) + '\nrows=[]\n'
            'for name in runs:\n'
            ' r=base/name;p=r/"progress.json";d=json.loads(p.read_bytes()) if p.exists() else {}\n'
            ' summary=r/"summary.json";completed=json.loads(summary.read_bytes()) if summary.exists() else []\n'
            ' rows.append(dict(name=name,progress=d,completed=len(completed)))\n'
            'checks=[base/runs[1]/(n+"/supplementary-summary.json") for n in ["scatter-history-1","scatter-history-2"]]\n'
            'print(json.dumps(dict(rows=rows,supplementary_complete=all(p.exists() for p in checks))))\n')
        status = remote(args.host, code)
        if status != previous:
            print(json.dumps(status), flush=True)
            previous = status
        if all(row['progress'].get('status') == 'completed' for row in status['rows']) and status['supplementary_complete']:
            break
        time.sleep(20)
    output = args.root.parent / 'runtime-output/presentation-exports-20261119'
    output.mkdir(parents=True, exist_ok=True)
    reports = args.root / 'development/reports'
    for name in RUNS:
        code = ('import json,subprocess\nfrom pathlib import Path\nbase=Path(' + repr(args.remote_base) + ')\n'
            'p=subprocess.run([' + repr(args.remote_python) + ',str(base/"presentation-replay-20261119/export.py"),'
            '"--run",str(base/' + repr(name) + '),"--output",str(base/(' + repr(name) + '+"-export.tar.gz"))],'
            'stdout=subprocess.PIPE,check=True)\nprint(p.stdout.decode())\n')
        exported = remote(args.host, code)
        local_archive = output / (name + '.tar.gz')
        subprocess.run(['scp', '-O', args.host + ':' + exported['archive'], str(local_archive)], check=True)
        assert hashlib.sha256(local_archive.read_bytes()).hexdigest() == exported['sha256']
        count = unpack_verified(local_archive, reports)
        print(json.dumps(dict(imported=name, files=count)), flush=True)
    for module in ['development.tools.traces.import_presentation_replay',
                   'development.tools.traces.build_group_meeting']:
        subprocess.run([sys.executable, '-m', module], cwd=args.root, check=True)
    sys.path.insert(0, str(args.root))
    from development.tools.traces.research_progress import write_report
    from development.tools.traces.progress_view import update_catalog
    for folder in ['qkv-simulator-20261118', 'native-upsample-contrast-20261118']:
        directory = reports / folder
        write_report(json.loads((directory / 'mainline-presentation.json').read_bytes()), directory)
    update_catalog(args.root, png=True)
    subprocess.run([sys.executable, '-m', 'development.tools.traces.check_presentation'], cwd=args.root, check=True)
    subprocess.run([sys.executable, '-m', 'development.tools.traces.meeting_slides'], cwd=args.root, check=True)
    print(json.dumps(dict(status='ready_for_screenshot_review')), flush=True)


if __name__ == '__main__':
    main()
