"""核验真实runtime事件与引用字节；旧可变路径可按原SHA解析，不改写旧记录。"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refs(value):
    if isinstance(value,dict):
        if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
            yield value
        for item in value.values():yield from refs(item)
    elif isinstance(value,list):
        for item in value:yield from refs(item)


def audit(run):
    run=Path(run).resolve();trace=run/'trace'
    documents=[]
    for folder in ['events','decisions','rounds']:
        for path in (trace/folder).rglob('*.json'):
            documents.append((path,json.loads(path.read_bytes())))
    # V1早期引用仍可能指向工作区，先寻找已在动作边界保存的同SHA原件。
    frozen={}
    for folder in ['sources','artifacts','evidence']:
        for path in (trace/folder).glob('*'):
            if path.is_file():frozen[sha(path)]=path
    for call in (run/'agent-calls').glob('*'):
        for name in ['input-source.py','output-source.py','input-diagnosis.json','initial-context.json']:
            path=call/name
            if path.is_file():frozen.setdefault(sha(path),path)
    checked={};missing=[];resolved=[];prefixes=[];reconstructed=[]
    for document,value in documents:
        for item in refs(value):
            key=(item['path'],item['sha256'])
            if key in checked:continue
            path=Path(item['path'])
            if not path.is_absolute():path=run/path
            raw=path.read_bytes() if path.is_file() else None
            if raw is not None and 'prefix_bytes' in item:
                raw=raw[:item['prefix_bytes']]
            if raw is not None and hashlib.sha256(raw).hexdigest()==item['sha256']:
                checked[key]='direct'
            elif item['sha256'] in frozen:
                checked[key]='same_sha_snapshot'
                resolved.append(dict(original=item['path'],sha256=item['sha256'],
                    snapshot=frozen[item['sha256']].relative_to(run).as_posix()))
            elif raw is not None and path.name=='agent-events.jsonl':
                # 早期记录没有长度。只在现存追加流的行边界寻找完全相同SHA，
                # 不修改日志、不猜测内容；不能匹配就明确保留缺口。
                digest=hashlib.sha256();length=0;found=None
                for line in raw.splitlines(keepends=True):
                    digest.update(line);length+=len(line)
                    if digest.hexdigest()==item['sha256']:found=length;break
                if found is not None:
                    checked[key]='verified_legacy_stream_prefix';prefixes.append(dict(**item,prefix_bytes=found))
                else:
                    checked[key]='unresolved';missing.append(dict(document=document.relative_to(run).as_posix(),**item))
            elif raw is not None and path.name=='source-provenance.json':
                value=json.loads(raw);value.pop('output_sha256',None)
                initial=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf8')
                if hashlib.sha256(initial).hexdigest()==item['sha256']:
                    checked[key]='verified_initial_metadata';reconstructed.append(dict(**item,method='去除后续追加的output_sha256并按原保存格式序列化，原SHA完全一致'))
                else:
                    checked[key]='unresolved';missing.append(dict(document=document.relative_to(run).as_posix(),**item))
            else:
                checked[key]='unresolved'
                missing.append(dict(document=document.relative_to(run).as_posix(),**item))
    events=[v for p,v in documents if p.parent.name=='events']
    started={v['id'] for v in events if v['kind']=='action_started'}
    ended=[v['started_event'] for v in events if v['kind']=='action_finished']
    orphan_ends=[key for key in ended if key not in started]
    open_actions=sorted(started-set(ended))
    decisions=[v for p,v in documents if p.parent.name=='decisions']
    return dict(schema='hgkda.runtime-trace-audit/v1',run=str(run),
        status='verified' if not missing and not orphan_ends and not open_actions else 'incomplete',
        events=len(events),decisions=len(decisions),unique_references=len(checked),
        source_snapshots=len(list((trace/'sources').glob('*.py'))),
        same_sha_resolutions=resolved,unresolved=missing,open_actions=open_actions,orphan_finishes=orphan_ends,
        verified_legacy_log_prefixes=prefixes,verified_initial_metadata=reconstructed,
        scope='验证事件闭合和引用字节身份；早期可变引用按已有同SHA快照解析。验证不证明模型因果归因或工具收益。')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.run_directory)
    with a.output.open('x',encoding='utf8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:result[k] for k in ['status','events','decisions','unique_references','source_snapshots']}))


if __name__=='__main__':main()
