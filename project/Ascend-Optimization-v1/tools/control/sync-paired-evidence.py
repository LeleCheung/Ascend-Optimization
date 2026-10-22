#!/usr/bin/env python3
"""同步成对批次证据：长路径原件保存在LFS归档，源码和短路径结果可直接阅读。"""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

REMOTE_PACK = r'''
from pathlib import Path
import hashlib,io,json,tarfile
root=Path('/data/hanle/ascend-optimization')
batch=root/'Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns'/parameters['batch']
target=root/'runs'/parameters['batch']/parameters['snapshot']
if target.exists():raise ValueError('snapshot already exists')
files=[]
for p in sorted(batch.rglob('*')):
 if not p.is_file():continue
 if p.is_symlink():raise ValueError('symlink evidence is not allowed')
 p.resolve().relative_to(batch.resolve())
 data=p.read_bytes()
 files.append((p,p.relative_to(batch).as_posix(),len(data),hashlib.sha256(data).hexdigest()))
if sum(x[2] for x in files)>512*1024*1024:raise ValueError('evidence exceeds 512 MiB')
manifest={'batch':batch.name,'format':'original-relative-paths/v1','files':[
 {'path':name,'size_bytes':size,'sha256':digest} for p,name,size,digest in files]}
payload=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
with tarfile.open(str(target),'w:gz') as archive:
 for p,name,size,digest in files:
  if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('evidence changed during snapshot')
  archive.add(str(p),arcname=name,recursive=False)
 item=tarfile.TarInfo('_snapshot-manifest.json');item.size=len(payload)
 archive.addfile(item,io.BytesIO(payload))
print(json.dumps({'path':str(target),'size_bytes':target.stat().st_size,
 'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'files':len(files)}))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',required=True,help='不可重用的归档名，例如 paired-final-20261022-v1.tar.gz')
    p.add_argument('--batch',default='paired-pointwise-20261022')
    p.add_argument('--allow-update',nargs='*',default=[],help='明确允许更新的项目相对证据；旧字节自动备份')
    a=p.parse_args()
    if PurePosixPath(a.snapshot).name!=a.snapshot or not a.snapshot.endswith('.tar.gz'):
        raise ValueError('非法归档名')
    if not a.batch.replace('-', '').replace('_', '').isalnum():
        raise ValueError('非法批次名')
    s=importlib.util.spec_from_file_location('paired_control',Path(__file__).with_name('control.py'))
    c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
    for name in a.allow_update:
        c.relative_name(name)
    meta=json.loads(c.remote_python('baai-910b',{'snapshot':a.snapshot,'batch':a.batch},REMOTE_PACK))
    if meta['size_bytes']>c.MAX_BYTES:raise ValueError('压缩归档超过256MiB')
    data=c.remote_python('baai-910b',{'path':meta['path']},
        "from pathlib import Path\nimport sys\nsys.stdout.buffer.write(Path(parameters['path']).read_bytes())\n")
    if len(data)!=meta['size_bytes'] or hashlib.sha256(data).hexdigest()!=meta['sha256']:
        raise ValueError('下载字节与远端回执不同')
    batch=c.PROJECT/'operators/campaigns'/a.batch
    pending=[]
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
        manifest=json.load(archive.extractfile('_snapshot-manifest.json'))
        by_name={r['path']:r for r in manifest['files']}
        seen=set()
        for entry in archive:
            if entry.name=='_snapshot-manifest.json':continue
            parts=c.relative_name(entry.name).parts
            if not entry.isfile() or entry.name in seen or entry.name not in by_name:
                raise ValueError('非法归档条目')
            seen.add(entry.name)
            content=archive.extractfile(entry).read();record=by_name[entry.name]
            if len(content)!=record['size_bytes'] or hashlib.sha256(content).hexdigest()!=record['sha256']:
                raise ValueError('归档内部SHA不符')
            target=batch.joinpath(*parts)
            # profiler内部长路径和二进制保留在原始归档；短路径源码、结果、比较JSON直接展开。
            if len(str(target))>235 or 'profile' in parts or 'discovery' in parts:
                continue
            target.resolve().relative_to(batch.resolve())
            old=target.read_bytes() if target.exists() else None
            if old is not None and old!=content:
                if (entry.name.endswith('-state.json') or '-state-' in target.name
                        or entry.name=='queue-state.json' or entry.name in a.allow_update):
                    pass
                else:raise ValueError('本地已有不同证据：'+entry.name)
            pending.append((target,content,old))
        if seen!=set(by_name):raise ValueError('归档丢失条目')
    for target,content,old in pending:
        if target.exists() and target.read_bytes()!=old:raise ValueError('写入前本地内容改变')
    for target,content,old in pending:
        if old is not None and old!=content:
            backup=c.REPO/'.control-state/download-backups'/hashlib.sha256(old).hexdigest()/target.relative_to(c.PROJECT)
            backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(old)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    destination=batch/'raw-evidence'/a.snapshot
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise ValueError('本地快照已存在')
    destination.write_bytes(data)
    (destination.parent/(a.snapshot+'.manifest.json')).write_text(
        json.dumps(dict(transfer=meta,contents=manifest),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(snapshot=str(destination),sha256=meta['sha256'],verified_files=len(seen),
                         materialized_files=len(pending)),ensure_ascii=False))


if __name__=='__main__':main()
