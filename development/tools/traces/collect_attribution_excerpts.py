"""Read bounded historical model logs; export simulator-related decision excerpts."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()
    root=Path('/data/hanle/ascend-optimization/runs')
    ops=['selu_','rad2deg_','log10_','uniform','negative_','native_group_norm','upsample_bilinear2d']
    records=[];profiles=[];visited=[]
    pattern=re.compile('ascend-instructions|WAIT_FLAG|BAR\\.ALL|instr_exe|regions\\.json|simulator.{0,35}(?:json|hotspot|instruction)|\u6a21\u62df\u5668',re.I)
    for ledger in root.glob('*/kg-runs/*/stages/optimize/work/.ledger.json'):
        op=next((o for o in ops if ledger.parents[3].name.startswith(o+'-')),None)
        if not op:continue
        for f in (ledger.parent/'.kernelgen/profile-analysis').glob('*.json'):
            raw=f.read_bytes();profiles.append({'operator':op,'path':str(f),'sha256':hashlib.sha256(raw).hexdigest(),'data':json.loads(raw)})
        for log in (ledger.parent/'.kernelgen/claude-config/projects').rglob('*.jsonl'):
            raw=log.read_bytes();identity={'operator':op,'path':str(log),'sha256':hashlib.sha256(raw).hexdigest()};visited.append(identity)
            blocks=[]
            for n,line in enumerate(raw.splitlines(),1):
                try:row=json.loads(line)
                except ValueError:continue
                message=row.get('message',{})
                if not isinstance(message,dict):continue
                content=message.get('content',[])
                if not isinstance(content,list):continue
                for b in content:
                    if not isinstance(b,dict):continue
                    kind=b.get('type')
                    if kind in ['text','thinking'] and row.get('type')=='assistant':
                        text=b.get('text',b.get('thinking',''))
                    elif kind=='tool_use':
                        args=b.get('input',{})
                        if b.get('name') in ['Edit','Write']:
                            text=json.dumps({'tool':b.get('name'),'id':b.get('id'),'file_path':args.get('file_path')})
                        else:text=json.dumps(b,ensure_ascii=False)
                    else:continue
                    # No environment dumps or credentials are exported.
                    text=re.sub(r'\b(?:sk-|ghp_)[A-Za-z0-9_-]{20,}','[REDACTED]',text)
                    blocks.append({'line':n,'type':kind,'time':row.get('timestamp'),'text':text[:10000]})
            hits=[i for i,b in enumerate(blocks) if pattern.search(b['text'])]
            selected=set(j for i in hits for j in range(max(0,i-2),min(len(blocks),i+9)))
            if hits:records.append(dict(identity,match_count=len(hits),excerpts=[blocks[i] for i in sorted(selected)]))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'scope':ops,'visited':visited,'profiles':profiles,'logs':records},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'logs':len(visited),'matched':len(records),'profiles':len(profiles)}))


if __name__=='__main__':main()
