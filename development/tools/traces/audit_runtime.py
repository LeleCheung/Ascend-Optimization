"""Audit runtime events and referenced bytes; resolve legacy mutable paths by original SHA without rewriting records."""
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
    # Early V1 references may target workspace paths; first locate same-SHA artifacts captured at action boundaries.
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
                # Legacy records lack prefix lengths. Search only line boundaries of existing append-only streams for an exact SHA;
                # preserve logs without guessing content; retain an explicit gap when no match exists.
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
                    checked[key]='verified_initial_metadata';reconstructed.append(dict(**item,method='Remove later output_sha256 and serialize in original format to reproduce the exact SHA'))
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
        scope='Verify event closure and referenced bytes; resolve legacy mutable references through existing same-SHA snapshots. This does not establish model causality or tool benefit.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.run_directory)
    with a.output.open('x',encoding='utf8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:result[k] for k in ['status','events','decisions','unique_references','source_snapshots']}))


if __name__=='__main__':main()
