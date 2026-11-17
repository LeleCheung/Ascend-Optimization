"""Recover hash-verified source transitions from successful literal Edit logs.

Only string substitution is replayed in memory. No candidate or logged command
is executed or written. A match is a reconstructed edit path, with log identity
and tool IDs retained for review, not an invented parent-round field.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    texts = {}
    for sub in ['export-v2/sources', 'links-v9/artifacts','extra-sources']:
        for f in (a.evidence/sub).glob('*.py'):
            raw=f.read_bytes()
            try:texts[sha(raw)] = raw.decode('utf8')
            except UnicodeError:pass
    logs = set()
    base = Path('/data/hanle/ascend-optimization/runs')
    for ledger in base.glob('*/kg-runs/*/stages/optimize/work/.ledger.json'):
        logs.update((ledger.parent/'.kernelgen/claude-config/projects').rglob('*.jsonl'))
    edges=[]; failures=[]
    for log in sorted(logs):
        tools={};results={}
        try:
            raw=log.read_bytes()
            for n,line in enumerate(raw.splitlines(),1):
                row=json.loads(line);message=row.get('message',{})
                if not isinstance(message,dict):continue
                content=message.get('content',[])
                if not isinstance(content,list):continue
                for block in content:
                    if not isinstance(block,dict):continue
                    if block.get('type')=='tool_use':tools[block['id']]=dict(block,line=n)
                    elif block.get('type')=='tool_result':results[block['tool_use_id']]=block
            states={};seen=set();confirmed={}
            for t in sorted(tools.values(),key=lambda t:t['line']):
                args=t.get('input',{});path=args.get('file_path','')
                result=results.get(t['id'])
                if 'finalize_round' in t.get('name','') and result and not result.get('is_error'):
                    content=result.get('content','')
                    if isinstance(content,list):content='\n'.join(v.get('text','') for v in content if isinstance(v,dict))
                    try:finalized=json.loads(content)
                    except (ValueError,TypeError):finalized={}
                    if not isinstance(finalized,dict) or finalized.get('candidate_action') not in ('KEEP','REVERT'):
                        continue
                    work=next((p.parent for p in log.parents if p.name=='.kernelgen'),None)
                    number=finalized.get('round_num') if finalized['candidate_action']=='KEEP' else finalized.get('best_round')
                    destination=finalized.get('candidate_path')
                    if work and isinstance(number,int) and destination:
                        snapshot=work/'.kernelgen/evals'/('round-'+str(number).zfill(4))/'main.py'
                        if snapshot.is_file():
                            value=snapshot.read_bytes();h=sha(value)
                            target=Path(destination)
                            if not target.is_absolute():target=work/target
                            if h in texts:
                                confirmed[str(target)]=(texts[h],h,[dict(tool_id=t['id'],line=t['line'],
                                    kind='controller_restored_evaluated_snapshot',source_sha256=h,
                                    action=finalized['candidate_action'],round=number,snapshot_path=str(snapshot))])
                                states.pop(str(target),None)
                    continue
                if any(name in t.get('name','') for name in ('eval_round','preflight_kernel')) and result and not result.get('is_error'):
                    content=result.get('content','')
                    if isinstance(content,list):content='\n'.join(v.get('text','') for v in content if isinstance(v,dict))
                    try:measurement=json.loads(content)
                    except (ValueError,TypeError):measurement={}
                    # kernel_sha256 identifies source bytes; solution_sha256 can
                    # identify the request bundle and must not replace it.
                    h=(measurement.get('kernel_sha256') or measurement.get('solution_sha256')) if isinstance(measurement,dict) else None
                    kernel_path=args.get('kernel_path')
                    if h in texts and kernel_path and (measurement.get('snapshot_path') or measurement.get('kernel_sha256')):
                        work=next((p.parent for p in log.parents if p.name=='.kernelgen'),None)
                        target=Path(kernel_path)
                        if not target.is_absolute() and work:target=work/target
                        confirmed[str(target)]=(texts[h],h,[dict(tool_id=t['id'],line=t['line'],
                            kind='evaluated_source_identity',source_sha256=h,snapshot_path=measurement.get('snapshot_path'))])
                    continue
                if Path(path).name not in ('main.py','candidate.py'):continue
                if not result or result.get('is_error'):continue
                if t.get('name')=='Read':
                    content=result.get('content','')
                    if isinstance(content,list):content='\n'.join(v.get('text','') for v in content if isinstance(v,dict))
                    if isinstance(content,str):
                        lines=[re.match(r'^\s*(\d+)\t(.*)$',line) for line in content.splitlines()]
                        numbered=[m for m in lines if m]
                        if numbered and [int(m[1]) for m in numbered]==list(range(1,len(numbered)+1)):
                            value='\n'.join(m[2] for m in numbered)
                            for trial in (value,value+'\n'):
                                h=sha(trial.encode())
                                if h in texts:confirmed[path]=(trial,h,[])
                    continue
                if t.get('name')=='Write':
                    value=args.get('content')
                    if not isinstance(value,str):continue
                    child=sha(value.encode())
                    previous=confirmed.get(path)
                    if previous:
                        _,parent,steps=previous
                        chain=steps+[dict(tool_id=t['id'],line=t['line'],kind='literal_write',content_sha256=child)]
                        if child in texts and parent!=child and (parent,child) not in seen:
                            seen.add((parent,child));edges.append(dict(parent_sha256=parent,source_sha256=child,
                                kind='observed_read_write_chain',log=str(log),log_sha256=sha(raw),file_path=path,steps=chain))
                    else:parent,chain=child,[]
                    confirmed[path]=(value,child,[]) if child in texts else (value,parent,chain)
                    states.pop(path,None)
                    continue
                if t.get('name')!='Edit':
                    continue
                before,after=args.get('old_string'),args.get('new_string')
                if not isinstance(before,str) or not before or not isinstance(after,str):continue
                candidates=dict(states.get(path,{}))
                for h,text in texts.items():candidates.setdefault(h,(text,h,[]))
                updated={}
                for _,(text,parent,steps) in candidates.items():
                    count=text.count(before)
                    if not count or (count!=1 and not args.get('replace_all',False)):continue
                    value=text.replace(before,after,-1 if args.get('replace_all') else 1)
                    child=sha(value.encode());chain=steps+[dict(tool_id=t['id'],line=t['line'],old_string_sha256=sha(before.encode()),new_string_sha256=sha(after.encode()))]
                    updated[child]=(value,parent,chain)
                    if child in texts and parent!=child and (parent,child) not in seen:
                        seen.add((parent,child))
                        edges.append(dict(parent_sha256=parent,source_sha256=child,kind='replayed_successful_literal_edits',
                                          log=str(log),log_sha256=sha(raw),file_path=path,steps=chain))
                states[path]=updated
                if path in confirmed:
                    text,parent,steps=confirmed[path];count=text.count(before)
                    if count and (count==1 or args.get('replace_all')):
                        value=text.replace(before,after,-1 if args.get('replace_all') else 1)
                        child=sha(value.encode());chain=steps+[dict(tool_id=t['id'],line=t['line'],kind='literal_edit')]
                        if child in texts and parent!=child and (parent,child) not in seen:
                            seen.add((parent,child));edges.append(dict(parent_sha256=parent,source_sha256=child,
                                kind='observed_read_edit_chain',log=str(log),log_sha256=sha(raw),file_path=path,steps=chain))
                        confirmed[path]=(value,child,[]) if child in texts else (value,parent,chain)
                    else:confirmed.pop(path,None)
        except (OSError,ValueError,TypeError) as exc:failures.append(dict(log=str(log),error=str(exc)))
    a.output.write_text(json.dumps(dict(logs=len(logs),edges=edges,errors=failures),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(logs=len(logs),edges=len(edges),errors=len(failures))))


if __name__=='__main__':main()
