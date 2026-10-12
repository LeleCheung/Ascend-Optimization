#!/usr/bin/env python3
"""通过 SSH 执行脚本、查看状态、上传项目文件或下载实验原始证据。"""
import argparse
import base64
import hashlib
import io
import json
import re
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT.parents[1]
OPERATORS = ('amin', 'matmul_bias_activation', 'narrow_copy')
MAX_BYTES = 256 * 1024 * 1024
DERIVED = {'最终候选.json', '精简汇报.md', '真机诊断.md',
           '版本对比.md', '版本对比.json', '版本对比.csv'}


def ssh_command(host, command):
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+', host) or host.startswith('-'):
        raise ValueError('请使用合法的 SSH 主机别名')
    return ['ssh', '-T', '-o', 'ConnectTimeout=20', host, command]


def remote_python(host, parameters, script):
    payload = 'parameters = ' + repr(parameters) + '\n' + script
    result = subprocess.run(ssh_command(host, 'python3 -'),
                            input=payload.encode('utf-8'), capture_output=True)
    if result.returncode:
        sys.stderr.write(result.stderr.decode('utf-8', errors='replace'))
        raise RuntimeError('远端命令失败，退出码 ' + str(result.returncode))
    if result.stdout.strip() == b'no available gateway':
        raise RuntimeError('堡垒机返回 no available gateway，未进入目标服务器')
    return result.stdout


def execute_script(host, path, container=None):
    command = 'bash -s'
    if container:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', container):
            raise ValueError('容器名称格式不正确')
        command = 'docker exec -i ' + shlex.quote(container) + ' bash -s'
    marker = b'CONTROL_EXEC_STARTED\n'
    script = path.read_text(encoding='utf-8-sig').replace('\r\n', '\n')
    payload = b"printf 'CONTROL_EXEC_STARTED\\n' >&2\n" + script.encode('utf-8')
    result = subprocess.run(ssh_command(host, command), input=payload, capture_output=True)
    if marker not in result.stderr:
        raise RuntimeError('目标 Bash 未确认启动：' + result.stdout.decode('utf-8', errors='replace').strip())
    sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr.replace(marker, b'', 1))
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, result.args)


def relative_name(name):
    path = PurePosixPath(name)
    if (not name or '\\' in name or path.is_absolute() or '..' in path.parts
            or name != path.as_posix() or ':' in name):
        raise ValueError('需要项目内的相对 POSIX 路径：' + name)
    return path


REMOTE_STATUS = r'''
from pathlib import Path
import json,subprocess,urllib.request
root=Path(parameters['root']).resolve()
project=root/'Ascend-Optimization/project/Ascend-Optimization-v1'
rows={'root':str(root),'sessions':[],'runs':[],'logs':[],'services':[],'results':[]}
tmux=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True)
rows['sessions']=[s for s in tmux.stdout.splitlines() if s.endswith(parameters['batch'])]
for f in sorted((root/'kg-runs').glob('*/.kernelgen/run-progress.json')):
 if parameters['batch'] not in f.parents[1].name:continue
 d=json.loads(f.read_text());rows['runs'].append({'workspace':f.parents[1].name,
   **{k:d.get(k) for k in ('state','stage','updated_at')}})
for f in sorted((root/'campaign-logs').glob('*.log')):
 lines=f.read_text(errors='replace').splitlines()
 rows['logs'].append({'name':f.name,'last_line':lines[-1][:600] if lines else ''})
for op in parameters['operators']:
 for f in sorted((project/'operators'/op/'reports').glob('**/*.result.json')):
  if parameters['batch'] not in f.relative_to(project).as_posix():continue
  d=json.loads(f.read_text());rows['results'].append({'path':str(f.relative_to(project)),
    **{k:d.get(k) for k in ('status','geo_mean','num_passed','num_workloads')}})
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for port in parameters['ports']:
 try:
  d=json.load(opener.open('http://127.0.0.1:'+str(port)+'/status',timeout=10))
  rows['services'].append({'port':port,'scheduler':d['scheduler']})
 except Exception as error:rows['services'].append({'port':port,'error':str(error)})
print(json.dumps(rows,ensure_ascii=False,indent=2))
'''

REMOTE_UPLOAD = r'''
from pathlib import Path
import base64,hashlib,json
root=Path(parameters['root']).resolve()
project=root/'Ascend-Optimization/project/Ascend-Optimization-v1'
assert project.is_dir(),project
changes=[]
for item in parameters['files']:
 p=project/item['name'];p.parent.resolve().relative_to(project.resolve())
 assert not p.is_symlink(),p
 assert not p.exists() or p.is_file(),p
 data=base64.b64decode(item['data'],validate=True)
 assert hashlib.sha256(data).hexdigest()==item['sha256'],p
 old=p.read_bytes() if p.exists() else None
 changes.append((p,data,old,item['name']))
for p,data,old,name in changes:
 if old==data:continue
 if old is not None:
  backup=root/'script-backups/control'/hashlib.sha256(old).hexdigest()/name
  backup.parent.mkdir(parents=True,exist_ok=True)
  assert not backup.exists() or backup.read_bytes()==old
  backup.write_bytes(old)
 p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
print(json.dumps({'checked':len(changes),'updated':sum(old!=data for p,data,old,name in changes)},ensure_ascii=False))
'''

REMOTE_DOWNLOAD = r'''
from pathlib import Path
import sys,tarfile,json
root=Path(parameters['root']).resolve()
project=root/'Ascend-Optimization/project/Ascend-Optimization-v1'
assert project.is_dir(),project
selected={}
def collect(directory,prefix,filter_batch=False):
 if not directory.is_dir():return
 for f in sorted(directory.rglob('*')):
  if not f.is_file() or f.is_symlink():continue
  f.resolve().relative_to(directory.resolve())
  name=f.relative_to(directory).as_posix()
  if filter_batch and parameters['batch'] not in name:continue
  if f.suffix=='.log' and not any(s in f.parts for s in ('artifacts','native-evidence','startup-failure')):continue
  selected.setdefault(prefix+name,f)
for op in parameters['operators']:
 directory=project/'operators'/op
 collect(directory/'candidates','operators/'+op+'/candidates/',True)
 collect(directory/'reports','operators/'+op+'/reports/',True)
for d in sorted((root/'diagnostics').glob('*')):
 if not d.is_dir() or parameters['batch'] not in d.name:continue
 op=('amin' if d.name.startswith('amin-') else 'matmul_bias_activation' if d.name.startswith('matmul-')
     else 'narrow_copy' if d.name.startswith('narrow-') else None)
 response=d/'response.json'
 if op not in parameters['operators'] or not response.is_file():continue
 if json.loads(response.read_text()).get('status')!='SUCCEEDED':continue
 collect(d,'operators/'+op+'/reports/diagnostics-'+parameters['batch']+'/'+d.name+'/')
assert selected,'未找到指定批次的证据'
assert sum(f.stat().st_size for f in selected.values())<=parameters['max_bytes'],'单批下载过大'
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name,f in sorted(selected.items()):archive.add(f,arcname=name,recursive=False)
'''


def download_actions(data, project, operators, batch, replace_derived=False):
    """先验证整个归档及冲突，再写入，避免半批覆盖。"""
    project = project.resolve()
    actions, seen, targets, total = [], set(), set(), 0
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        for member in archive:
            parts = relative_name(member.name).parts
            if (not member.isfile() or len(parts) < 4 or parts[0] != 'operators'
                    or parts[1] not in operators or parts[2] not in ('candidates', 'reports')
                    or batch not in '/'.join(parts[3:]) or member.name in seen):
                raise ValueError('非法或重复的归档条目：' + member.name)
            seen.add(member.name)
            total += member.size
            if member.size < 0 or total > MAX_BYTES:
                raise ValueError('归档超过下载大小限制')
            target = project.joinpath(*parts)
            target.resolve().relative_to(project)
            if target.resolve() in targets:
                raise ValueError('多个条目指向同一本地路径：' + member.name)
            targets.add(target.resolve())
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise ValueError('目标不是普通文件：' + member.name)
            content = archive.extractfile(member).read()
            old = target.read_bytes() if target.exists() else None
            if old == content:
                continue
            if old is not None:
                if old.replace(b'\r\n', b'\n') == content and committed_checkout(target, content, project):
                    continue
                derived = ('master-closure-' + batch in parts and target.name in DERIVED)
                if not (replace_derived and derived):
                    raise ValueError('已有文件字节不同，拒绝覆盖：' + member.name)
            actions.append((target, content, old))
    if not seen:
        raise ValueError('归档为空')
    return actions, len(seen)


def committed_checkout(target, content, project):
    """仅认可与提交 blob 相同的换行差异；保留本地证据字节。"""
    repo = project.parents[1]
    if not (repo / '.git').exists():
        return False
    name = target.relative_to(repo).as_posix()
    result = subprocess.run(['git', '-C', str(repo), 'show', 'HEAD:' + name], capture_output=True)
    return result.returncode == 0 and result.stdout == content


def apply_download(actions, project, state):
    for path, data, old in actions:
        current = path.read_bytes() if path.exists() else None
        if current != old:
            raise ValueError('下载检查后本地文件已改变：' + str(path))
    for path, data, old in actions:
        if old is not None:
            backup = state / hashlib.sha256(old).hexdigest() / path.relative_to(project)
            backup.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists() and backup.read_bytes() != old:
                raise ValueError('备份字节冲突：' + str(backup))
            backup.write_bytes(old)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='baai-910b', help='已配置的 SSH 主机别名')
    parser.add_argument('--remote-root', help='910B 实验根目录，包含 Ascend-Optimization 与 kg-runs')
    commands = parser.add_subparsers(dest='command', required=True)
    execute = commands.add_parser('exec', help='执行本地 Bash 文件，去 BOM 并转为 LF')
    execute.add_argument('script', type=Path)
    execute.add_argument('--container', help='指定后在该容器内执行')
    upload = commands.add_parser('upload', help='上传明确指定的项目文件，远端旧内容自动备份')
    upload.add_argument('files', nargs='+', help='相对 project/Ascend-Optimization-v1 的路径')
    for name in ('status', 'download'):
        cmd = commands.add_parser(name, help='查看实验状态' if name == 'status' else '下载原始证据')
        cmd.add_argument('--batch', required=True, help='批次后缀，例如 20261010')
        cmd.add_argument('--operators', nargs='+', choices=OPERATORS, default=list(OPERATORS))
        if name == 'status':
            cmd.add_argument('--ports', nargs='+', type=int, default=[19655, 19657])
        else:
            cmd.add_argument('--replace-derived', action='store_true', help='备份后替换明确的派生汇总文件')
    args = parser.parse_args()
    if args.command == 'exec':
        execute_script(args.host, args.script, args.container)
        return
    if not args.remote_root or not PurePosixPath(args.remote_root).is_absolute():
        parser.error('需要 --remote-root 指定远端绝对实验路径')
    if args.command == 'upload':
        files = []
        for name in args.files:
            path = PROJECT.joinpath(*relative_name(name).parts)
            path.resolve().relative_to(PROJECT)
            if path.is_symlink() or not path.is_file():
                raise ValueError('上传对象必须是项目内普通文件：' + name)
            data = path.read_bytes()
            files.append({'name': name, 'sha256': hashlib.sha256(data).hexdigest(),
                          'data': base64.b64encode(data).decode('ascii')})
        if len({f['name'] for f in files}) != len(files):
            parser.error('上传文件重复')
        if sum(len(f['data']) for f in files) > MAX_BYTES:
            parser.error('单批上传过大')
        result = json.loads(remote_python(args.host, {'root': args.remote_root, 'files': files}, REMOTE_UPLOAD))
        if result.get('checked') != len(files):
            raise ValueError('上传回执与文件数不一致')
        print(json.dumps(result, ensure_ascii=False))
        return
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.batch):
        parser.error('批次仅允许字母、数字、横线和下划线')
    parameters = {'root': args.remote_root, 'batch': args.batch, 'operators': args.operators}
    if args.command == 'status':
        if any(not 1 <= port <= 65535 for port in args.ports):
            parser.error('端口超出范围')
        parameters['ports'] = args.ports
        result = json.loads(remote_python(args.host, parameters, REMOTE_STATUS))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        parameters['max_bytes'] = MAX_BYTES
        data = remote_python(args.host, parameters, REMOTE_DOWNLOAD)
        actions, total = download_actions(data, PROJECT, args.operators, args.batch, args.replace_derived)
        apply_download(actions, PROJECT, REPO / '.control-state/download-backups')
        print(json.dumps({'checked': total, 'written': len(actions)}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
