"""通过SSH标准输入同步本批文件，逐文件核验SHA，避免依赖堡垒机SFTP。"""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

PROJECT=Path(__file__).resolve().parents[4]
REMOTE='/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1'


def io_path(path):
    path=Path(path)
    return Path('\\\\?\\'+str(path.resolve())) if os.name=='nt' and not str(path).startswith('\\\\?\\') else path


def execute(code):
    result=subprocess.run(['ssh','-o','BatchMode=yes','baai-910b','python3 -'],input=code.encode('utf8'),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=300)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf8',errors='replace')+'\n'+result.stdout.decode('utf8',errors='replace')[-3000:])
    return result.stdout


def push(names):
    items=[]
    for name in names:
        path=(PROJECT/name).resolve();path.relative_to(PROJECT)
        data=path.read_bytes()
        items.append(dict(path=path.relative_to(PROJECT).as_posix(),data=base64.b64encode(data).decode(),sha256=hashlib.sha256(data).hexdigest()))
    code='import base64,hashlib,json\nfrom pathlib import Path\nroot=Path('+repr(REMOTE)+')\nitems='+repr(items)+'''\nfor item in items:
    path=(root/item['path']).resolve();path.relative_to(root)
    raw=base64.b64decode(item['data']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.upload-tmp')
    temporary.write_bytes(raw);temporary.replace(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
print(json.dumps(dict(status='verified',files=len(items))))
'''
    print(execute(code).decode().strip())


def pull(batch):
    if not batch.replace('-','').isalnum():raise ValueError('非法批次名')
    relative='operators/campaigns/'+batch
    code='import base64,hashlib,io,json,tarfile\nfrom pathlib import Path\nroot=Path('+repr(REMOTE)+')\nfolder=root/'+repr(relative)+'''\nmanifest={};buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
    for path in sorted(folder.rglob('*')):
        if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts and 'compiler-cache' not in path.parts:
            raw=path.read_bytes();name=path.relative_to(root).as_posix()
            manifest[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
            item=tarfile.TarInfo(name);item.size=len(raw);archive.addfile(item,io.BytesIO(raw))
print(json.dumps(dict(files=manifest,archive=base64.b64encode(buffer.getvalue()).decode())))
'''
    value=json.loads(execute(code))
    archive=io.BytesIO(base64.b64decode(value['archive']))
    with tarfile.open(fileobj=archive,mode='r:gz') as stream:
        for member in stream:
            if not member.isfile():raise ValueError('不允许非普通文件')
            target=(PROJECT/member.name).resolve();target.relative_to(PROJECT)
            raw=stream.extractfile(member).read();expected=value['files'][member.name]
            assert hashlib.sha256(raw).hexdigest()==expected['sha256']
            target=io_path(target)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(raw)
    for name,expected in value['files'].items():
        assert hashlib.sha256(io_path(PROJECT/name).read_bytes()).hexdigest()==expected['sha256']
    manifest=PROJECT/'operators/campaigns'/(batch+'-windows-sync.json')
    manifest.write_text(json.dumps(dict(status='verified',files=value['files']),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='verified',files=len(value['files']),manifest=str(manifest))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    sub.add_parser('push').add_argument('paths',nargs='+')
    sub.add_parser('pull').add_argument('--batch',required=True)
    args=parser.parse_args()
    push(args.paths) if args.mode=='push' else pull(args.batch)
