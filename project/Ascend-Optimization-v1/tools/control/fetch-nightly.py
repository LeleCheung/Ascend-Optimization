#!/usr/bin/env python3
"""下载夜间批次证据，核验整批SHA后写入；运行中仅允许下载到本地临时快照。"""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import tarfile

spec=importlib.util.spec_from_file_location('control',Path(__file__).with_name('control.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
REMOTE=r'''
import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(parameters['root'])/'Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns'/parameters['batch']
state=root/'queue-state.json'
assert root.is_dir(),'批次目录不存在'
interruption=root/'interruption.json'
stopped=json.loads(interruption.read_text()) if interruption.is_file() else {}
audited_stop=(state.is_file() and stopped.get('status')=='stopped' and
              stopped.get('queue_process_absent') is True and stopped.get('tmux_absent') is True and
              stopped.get('unanswered_requests')==[] and
              stopped.get('queue_state_sha256')==hashlib.sha256(state.read_bytes()).hexdigest())
assert parameters['snapshot'] or audited_stop or (state.is_file() and json.loads(state.read_text())['status'] in ['completed','needs_attention']),'运行中或排队中只能下载临时快照'
files={};total=0
for path in sorted(root.rglob('*')):
 if not path.is_file() or path.is_symlink() or '__pycache__' in path.parts or path.suffix in ['.pyc','.pyo']:continue
 before=path.stat();data=path.read_bytes();after=path.stat()
 assert (before.st_mtime_ns,before.st_size)==(after.st_mtime_ns,after.st_size),'读取时文件仍在改变，请稍后重试'
 total+=len(data);assert total<=512*1024*1024,'本批超过512MiB，请分批归档'
 files[path.relative_to(root).as_posix()]=data
manifest={'batch':parameters['batch'],'files':{n:hashlib.sha256(d).hexdigest() for n,d in files.items()}}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name,data in [('_transfer-manifest.json',json.dumps(manifest).encode())]+list(files.items()):
  member=tarfile.TarInfo(name);member.size=len(data);archive.addfile(member,io.BytesIO(data))
'''


def fetch(host,root,batch,snapshot):
    data=c.remote_python(host,{'root':root,'batch':batch,'snapshot':snapshot},REMOTE)
    contents={};total=0
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
        for member in archive:
            c.relative_name(member.name)
            if not member.isfile() or member.name in contents:raise ValueError('重复或非常规归档条目')
            total+=member.size
            if member.size<0 or total>513*1024*1024:raise ValueError('归档过大')
            contents[member.name]=archive.extractfile(member).read()
    manifest=json.loads(contents.pop('_transfer-manifest.json'))
    if manifest['batch']!=batch or manifest['files']!={n:hashlib.sha256(d).hexdigest() for n,d in contents.items()}:
        raise ValueError('传输清单与下载字节不一致')
    target=(c.REPO/'.control-state/nightly-snapshots'/batch if snapshot else c.PROJECT/'operators/campaigns'/batch)
    actions=[]
    mutable={'queue-state.json','进展.md','精简汇报.md','统计核验.json','结果汇总.md','交付清单.md','交付清单.json','finalization.json','postflight.json','repair-verification.json','reference-execution-audit.json',
             '版本对比.md','版本对比.json','版本对比.csv'}
    for name,content in contents.items():
        path=target.joinpath(*c.relative_name(name).parts)
        path.resolve().relative_to(target.resolve())
        if path.is_symlink():raise ValueError('本地目标不能是符号链接')
        old=path.read_bytes() if path.exists() else None
        if old==content:continue
        if old is not None and not snapshot and path.name not in mutable:
            raise ValueError('已有冻结证据不同，拒绝覆盖: '+name)
        actions.append((path,content,old))
    c.apply_download(actions,target,c.REPO/'.control-state/nightly-fetch-backups'/batch)
    return {'files':len(contents),'written':len(actions),'directory':str(target),'snapshot':snapshot}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='baai-910b')
    parser.add_argument('--remote-root',default='/data/hanle/ascend-optimization')
    parser.add_argument('--batch',required=True)
    parser.add_argument('--snapshot',action='store_true',help='运行中下载到被Git忽略的本地快照目录')
    args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]+',args.batch):parser.error('非法批次名')
    print(json.dumps(fetch(args.host,args.remote_root,args.batch,args.snapshot),ensure_ascii=False))
