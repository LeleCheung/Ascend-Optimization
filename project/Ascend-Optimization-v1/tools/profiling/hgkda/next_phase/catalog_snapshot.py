"""为KG冻结不含派生字节码的catalog；原目录与所有合同原字节保持原样。"""
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_catalog(source, destination, records):
    source=Path(source).resolve();destination=Path(destination).resolve()
    for record in records:
        relative=Path(record['catalog_leaf']).relative_to('catalog')
        leaf=source/relative
        for name,expected in record['contract_sha256'].items():
            if digest(leaf/name)!=expected:
                raise ValueError('冻结合同发生变化：'+str(leaf/name))
    files={};excluded=[]
    for path in sorted(source.rglob('*')):
        relative=path.relative_to(source)
        if path.is_symlink():raise ValueError('catalog快照拒绝符号链接：'+str(path))
        if '__pycache__' in relative.parts:
            if path.is_file():
                if path.suffix!='.pyc':raise ValueError('缓存目录含非字节码文件：'+str(path))
                excluded.append(str(relative))
            continue
        if path.is_file():files[str(relative)]=digest(path)
    if not destination.exists():
        shutil.copytree(source,destination,ignore=shutil.ignore_patterns('__pycache__'))
    actual={str(path.relative_to(destination)):digest(path)
            for path in destination.rglob('*') if path.is_file()}
    if files!=actual:raise ValueError('已有catalog快照与冻结输入不同，拒绝覆盖')
    result=dict(status='verified',source=str(source),destination=str(destination),
                files=files,excluded_generated_bytecode=excluded,
                scope='只排除__pycache__内的派生pyc；所有其他文件原字节复制，合同逐项核对SHA')
    destination.with_name(destination.name+'-provenance.json').write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return destination
