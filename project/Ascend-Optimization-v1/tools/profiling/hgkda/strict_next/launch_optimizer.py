"""一次性部署持续搜索；先确认已有采集和独立提案进程真正退出。"""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


ROOT = Path('/data/hanle/ascend-optimization')
PROJECT = ROOT / 'Ascend-Optimization/project/Ascend-Optimization-v1'
BATCH = PROJECT / 'operators/campaigns/strict-next-20261103'
TOOLS = PROJECT / 'tools/profiling/hgkda/strict_next'


def read(path):
    return json.loads(path.read_text())


def main():
    assert not (BATCH / 'optimizer-process.json').exists(), '已有优化器身份，不能重复启动'
    assert not (BATCH / 'optimizer-launch.json').exists(), '已登记启动，须检查实际身份'
    for name in ['collector', 'fma-evaluator', 'typed-fma-evaluator']:
        value = read(BATCH / (name + '-process.json'))
        assert value['status'] in ['completed', 'requires_review']
        proc = Path('/proc') / str(value['pid']) / 'stat'
        assert not proc.exists() or proc.read_text().rsplit(')', 1)[1].split()[19] != str(value['process_start'])
    assert read(BATCH / 'typed-fma-evaluator-process.json')['status'] == 'completed'
    assert read(BATCH / 'proposals/expm1_/codex-typed-fma-v2/result.json')['status'] == 'evaluated'
    allowed = {'tools/profiling/hgkda/strict_next/' + name for name in ['optimize.py', 'search.py']}
    with tarfile.open('/tmp/hanle-strict-next-optimizer-transfer.tar.gz') as archive:
        members = archive.getmembers()
        assert {m.name for m in members} == allowed
        for member in members:
            assert member.isfile() and not (PROJECT / member.name).exists()
            destination = (PROJECT / member.name).resolve()
            assert str(destination).startswith(str(PROJECT.resolve()) + '/')
            destination.write_bytes(archive.extractfile(member).read())
    methods = {}
    for path in sorted(TOOLS.glob('*.py')):
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        frozen = BATCH / 'method-snapshots' / digest / path.name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        if frozen.exists():
            assert frozen.read_bytes() == data
        else:
            frozen.write_bytes(data)
        methods[path.name] = digest
    session = 'hanle-strict-next-20261103'
    exists = subprocess.run(['tmux', 'has-session', '-t', session], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if exists.returncode:
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-n', 'optimizer-ready'], check=True)
    log = ROOT / 'runs/strict-next-20261103/optimizer.log'
    command = 'exec ' + ' '.join(shlex.quote(str(v)) for v in [ROOT / 'runtime/kg-controller/venv/bin/python', TOOLS / 'optimize.py'])
    command += ' >' + shlex.quote(str(log)) + ' 2>&1'
    value = dict(method_sha256=methods, session=session, command=command,
                 planned_searches=2, rounds_per_search=5, engine_budget_seconds_per_search=2400)
    with (BATCH / 'optimizer-launch.json').open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    result = subprocess.run(['tmux', 'new-window', '-d', '-t', session, '-n', 'continuous-search', '-P', '-F', '#{window_id} #{pane_pid}', command],
                            check=True, stdout=subprocess.PIPE, universal_newlines=True)
    print(result.stdout.strip())


if __name__ == '__main__':
    main()
