"""由Git提交生成实体文件SHA清单；LFS按实体OID核验，不把指针当成复现文件。"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def manifest(repo,revision='HEAD'):
    repo=Path(repo)
    commit=subprocess.check_output(['git','rev-parse',revision],cwd=str(repo)).decode().strip()
    tree=subprocess.check_output(['git','ls-tree','-rz',commit],cwd=str(repo))
    process=subprocess.Popen(['git','cat-file','--batch'],cwd=str(repo),stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    cache={};files={}
    try:
        for entry in tree.split(b'\0'):
            if not entry:continue
            meta,name=entry.split(b'\t',1);mode,kind,oid=meta.decode().split()
            if kind!='blob':raise ValueError('非普通blob需单独核验：'+name.decode('utf8'))
            if oid not in cache:
                process.stdin.write((oid+'\n').encode());process.stdin.flush()
                header=process.stdout.readline().decode().split();assert header[:2]==[oid,'blob']
                length=int(header[2]);data=process.stdout.read(length);assert len(data)==length and process.stdout.read(1)==b'\n'
                lfs=data.startswith(b'version https://git-lfs.github.com/spec/v1\n')
                if lfs:
                    digest=re.search(rb'^oid sha256:([a-f0-9]{64})$',data,re.M)
                    size=re.search(rb'^size ([0-9]+)$',data,re.M);assert digest and size
                    identity=dict(sha256=digest.group(1).decode(),bytes=int(size.group(1)),lfs=True)
                else:identity=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),lfs=False)
                cache[oid]=identity
            files[name.decode('utf8')]=dict(cache[oid],mode=mode,git_blob=oid)
    finally:
        process.stdin.close();process.stdout.close();assert process.wait()==0
    return dict(schema='hgkda-repository-entity-identity-v1',commit=commit,files=files,
        scope='提交中全部受版本控制文件；LFS指针展开为实体SHA和字节数。未跟踪运行产物不删除、不计入相同文件集合。')


def compare(root,value):
    root=Path(root).resolve();different=[];missing=[];verified=0
    for name,expected in value['files'].items():
        path=root/name
        if os.name=='nt' and not str(path).startswith('\\\\?\\'):path=Path('\\\\?\\'+str(path))
        if expected['mode']=='120000':
            if not path.is_symlink():different.append(dict(path=name,reason='symlink_missing'));continue
            raw=os.readlink(path).encode();actual=hashlib.sha256(raw).hexdigest();size=len(raw)
        else:
            if not path.is_file():missing.append(name);continue
            size=path.stat().st_size
            if size!=expected['bytes']:
                different.append(dict(path=name,reason='size',expected=expected['bytes'],actual=size));continue
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
            actual=digest.hexdigest()
        if actual!=expected['sha256']:different.append(dict(path=name,reason='sha256',expected=expected['sha256'],actual=actual))
        else:verified+=1
    return dict(status='verified' if not different and not missing else 'different',commit=value['commit'],
        total=len(value['files']),verified=verified,missing=missing,different=different)


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    build=s.add_parser('manifest');build.add_argument('--repo',type=Path,required=True);build.add_argument('--revision',default='HEAD');build.add_argument('--output',type=Path,required=True)
    check=s.add_parser('compare');check.add_argument('--root',type=Path,required=True);check.add_argument('--manifest',type=Path,required=True);check.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='manifest':
        value=manifest(a.repo,a.revision);raw=json.dumps(value,ensure_ascii=False,separators=(',',':')).encode('utf8')
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(gzip.compress(raw,mtime=0))
        print(json.dumps(dict(commit=value['commit'],files=len(value['files']),lfs=sum(v['lfs'] for v in value['files'].values()))))
    else:
        value=json.loads(gzip.decompress(a.manifest.read_bytes()));result=compare(a.root,value)
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf8'))
        print(json.dumps(dict(status=result['status'],total=result['total'],verified=result['verified'],missing=len(result['missing']),different=len(result['different']))))


if __name__=='__main__':main()
