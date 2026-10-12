#!/usr/bin/env python3
"""本次 910B 目录迁移：逐阶段执行，删除前保存独有字节并验证归档。"""
import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tarfile
import time
from pathlib import Path

ROOT = Path('/data/hanle/ascend-optimization')
JOURNAL = ROOT / 'runs/migration-20261012'
CONTAINER = 'tle_yy'
PORTS = (19650, 19652, 19653, 19654, 19655, 19656, 19657, 19658)


def save(name, data):
    JOURNAL.mkdir(parents=True, exist_ok=True)
    (JOURNAL / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def run(*args, **kwargs):
    return subprocess.check_output(list(args), **kwargs)


def bound(path):
    path.absolute().relative_to(ROOT)
    if path == ROOT or path.is_symlink():
        raise ValueError('拒绝操作根目录或符号链接：' + str(path))
    path.resolve().relative_to(ROOT.resolve())
    return path


def process(pid):
    base = Path('/proc') / str(pid)
    try:
        cmd = (base / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        start = (base / 'stat').read_text().rsplit(')', 1)[1].split()[19]
        env = dict(v.split('=', 1) for v in (base / 'environ').read_bytes().decode().split('\0') if '=' in v)
        return {'pid': int(pid), 'command': cmd, 'start': start,
                'environment': {k: env.get(k) for k in ('PYTHONPATH', 'KGS_FLAGGEMS_ROOT', 'ASCEND_RT_VISIBLE_DEVICES')}}
    except (OSError, UnicodeError):
        return None


def status():
    code = '''import json,urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
rows=[]
for port in %r:
 d=json.load(opener.open('http://127.0.0.1:'+str(port)+'/status',timeout=10))
 rows.append({'port':port,'scheduler':d['scheduler']})
print(json.dumps(rows))
''' % (PORTS,)
    return json.loads(run('docker', 'exec', '-i', CONTAINER, 'python3', '-', input=code.encode()))


def audit():
    rows = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        row = process(p.name)
        if row and any(s in row['command'] for s in ('-m kernelgen_server.server', 'from kernelgen_server.server import main')):
            if str(ROOT) in row['command']:
                rows.append(row)
    if len(rows) != len(PORTS):
        raise RuntimeError('项目服务数量发生变化，先重新审阅：' + str(len(rows)))
    queues = status()
    if any(r['scheduler']['active'] or r['scheduler']['waiting'] for r in queues):
        raise RuntimeError('仍有评测排队或执行，不能迁移')
    save('before.json', {'services': rows, 'queues': queues,
                         'top_level': sorted(p.name for p in ROOT.iterdir())})
    print('AUDIT_IDLE', len(rows), flush=True)


def stop():
    before = json.loads((JOURNAL / 'before.json').read_text())
    queues = status()
    if any(r['scheduler']['active'] or r['scheduler']['waiting'] for r in queues):
        raise RuntimeError('队列不为空，拒绝停止')
    for row in before['services']:
        now = process(row['pid'])
        if now != row:
            raise RuntimeError('服务进程身份发生变化：' + str(row['pid']))
    for row in before['services']:
        os.kill(row['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not any(process(r['pid']) for r in before['services']):
            break
        time.sleep(1)
    else:
        raise RuntimeError('进程尚未退出；未强制杀死，请核对')
    save('stopped.json', {'pids': [r['pid'] for r in before['services']], 'queues': queues})
    print('STOPPED_PROJECT_SERVICES', len(before['services']), flush=True)


def move(source, target):
    bound(source); bound(target)
    if not source.exists():
        raise RuntimeError('待迁移目录不存在：' + str(source))
    if target.exists():
        raise RuntimeError('目标已存在，拒绝覆盖：' + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)
    print('MOVED', str(source.relative_to(ROOT)), str(target.relative_to(ROOT)), flush=True)


def relocate():
    assert (JOURNAL / 'stopped.json').is_file()
    move(ROOT / 'Ascend-Optimization', JOURNAL / 'old-Ascend-Optimization')
    move(ROOT / 'goal-20261010/Ascend-Optimization', ROOT / 'Ascend-Optimization')
    move(ROOT / 'goal-20261010', ROOT / 'runs/20261010')
    controller = ROOT / 'runtime/kg-controller'
    move(controller / 'runs', ROOT / 'runs/legacy-pre-master')
    for name in ('catalog-square', 'catalog-softmax', 'catalog-native-layer-norm', 'definitions'):
        if (controller / name).exists():
            move(controller / name, ROOT / 'runtime/catalogs' / name)
    # 控制器可执行文件、venv 和 Claude 保持路径，早期脚本与日志按批次保存。
    keep = {'venv', 'bin', 'bin-claude', 'claude', 'claude-native', 'tooling'}
    for p in sorted(controller.iterdir()):
        if p.name not in keep:
            move(p, ROOT / 'runs/legacy-pre-master/controller-files' / p.name)
    for p in sorted((ROOT / 'runtime').iterdir()):
        if p.name not in {'kg-controller', 'catalogs', 'kg-dedicated-deps'}:
            move(p, ROOT / 'runs/legacy-pre-master/runtime-files' / p.name)
    for p in sorted(ROOT.iterdir()):
        if p.is_file():
            move(p, JOURNAL / 'loose-files' / p.name)
    save('relocated.json', {'repo': str(ROOT / 'Ascend-Optimization'), 'batch': str(ROOT / 'runs/20261010')})


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def files(directory):
    for base, dirs, names in os.walk(directory, followlinks=False):
        dirs[:] = [n for n in dirs if n not in {'.git', '__pycache__', '.pytest_cache'}]
        for name in names:
            if not name.endswith('.pyc'):
                yield Path(base) / name
        for name in list(dirs):
            p = Path(base) / name
            if p.is_symlink():
                dirs.remove(name)
                yield p


def compact():
    assert (JOURNAL / 'validated.json').is_file(), '先完成正式服务和三算子复验'
    repo = ROOT / 'Ascend-Optimization'
    targets = [JOURNAL / 'old-Ascend-Optimization', ROOT / 'Ascend-Optimization-current',
               ROOT / 'KernelGen', ROOT / 'FlagGems', ROOT / 'sources']
    targets += [ROOT / 'runs/20261010' / n for n in ('FlagGems', 'FlagGems-case-adapter',
                 'FlagGems-narrow-device-timing', 'FlagGems-narrow-operator-timing')]
    targets.append(ROOT / 'runs/legacy-pre-master/runtime-files/flagggems-narrow-copy')
    # 只清理列表内的旧副本；每个独有文件先写入归档并逐字节校验。
    index = {}
    for p in files(repo):
        if p.is_symlink():
            continue
        index.setdefault(p.stat().st_size, []).append(p)
    cache = {}
    rows = []
    archive_path = JOURNAL / 'legacy-unique-files.tar.gz'
    if archive_path.exists():
        raise RuntimeError('归档已经存在，拒绝重复清理')
    with tarfile.open(archive_path, 'w:gz') as archive:
        for target in targets:
            bound(target)
            if not target.exists():
                continue
            if (target / '.git').exists():
                name = target.name + '-' + hashlib.sha256(str(target).encode()).hexdigest()[:8]
                bundle = JOURNAL / (name + '.bundle')
                run('git', '-C', str(target), 'bundle', 'create', str(bundle), '--all')
                run('git', '-C', str(target), 'bundle', 'verify', str(bundle), stderr=subprocess.DEVNULL)
            for p in files(target):
                name = str(p.relative_to(ROOT))
                if p.is_symlink():
                    archive.add(p, arcname=name, recursive=False)
                    rows.append({'path': name, 'link': os.readlink(p), 'storage': 'archive'})
                    continue
                sha = digest(p)
                duplicate = None
                for q in index.get(p.stat().st_size, []):
                    if q not in cache:
                        cache[q] = digest(q)
                    if sha == cache[q]:
                        duplicate = str(q.relative_to(ROOT))
                        break
                row = {'path': name, 'sha256': sha, 'size': p.stat().st_size}
                if duplicate:
                    row.update(storage='repository', preserved_path=duplicate)
                else:
                    archive.add(p, arcname=name, recursive=False)
                    row['storage'] = 'archive'
                rows.append(row)
    with tarfile.open(archive_path, 'r:gz') as archive:
        for row in rows:
            if row['storage'] != 'archive':
                if digest(ROOT / row['preserved_path']) != row['sha256']:
                    raise RuntimeError('仓库中的保留文件发生变化')
            elif 'link' in row:
                if archive.getmember(row['path']).linkname != row['link']:
                    raise RuntimeError('符号链接归档不一致')
            elif hashlib.sha256(archive.extractfile(row['path']).read()).hexdigest() != row['sha256']:
                raise RuntimeError('独有文件归档不一致')
    save('legacy-files.json', {'files': rows, 'archive_sha256': digest(archive_path)})
    for target in targets:
        if target.exists():
            shutil.rmtree(bound(target))
            print('REMOVED_VERIFIED_COPY', str(target.relative_to(ROOT)), flush=True)
    save('compacted.json', {'files_checked': len(rows), 'unique_files': sum(r['storage'] == 'archive' for r in rows)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('audit', 'stop', 'relocate', 'compact'))
    args = parser.parse_args()
    {'audit': audit, 'stop': stop, 'relocate': relocate, 'compact': compact}[args.phase]()


if __name__ == '__main__':
    main()
