"""Build a live, source-linked inventory without converting failed attempts into scores."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);a=p.parse_args();r=a.study
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    baseline=json.loads((r/'native-result.json').read_bytes())
    result={'schema':'hgkda.native-contrast-inventory/v1','upstream':json.loads((r/'experiment.json').read_bytes()),
            'baseline':{k:baseline.get(k) for k in ['status','num_passed','num_workloads','geo_mean']},'arms':{}}
    for arm in ['no-simulator','with-simulator','compatibility']:
        run=r/arm;rows=[]
        for call in sorted((run/'agent-calls').glob('*')):
            if not call.is_dir():continue
            source=call/'input-source.py';output=call/'output-source.py'
            row={'round':call.name,'input_sha256':sha(source),'output_sha256':sha(output) if output.exists() else None,
                 'decisions':[],'simulator_reads':[],'external_file_reads':[],'candidate_edits':0}
            events=call/'agent-events.jsonl'
            if events.exists():
                row['model_stream_sha256']=sha(events);read_ids={};success=set();first_edit=None;decision_request=None
                for order,line in enumerate(events.read_bytes().splitlines()):
                    try:e=json.loads(line)
                    except json.JSONDecodeError:continue
                    if e.get('type')=='result':row['model_result']={k:e.get(k) for k in ['subtype','num_turns','duration_ms','total_cost_usd','usage']}
                    msg=e.get('message',{})
                    if not isinstance(msg,dict):continue
                    for b in msg.get('content',[]):
                        if not isinstance(b,dict):continue
                        if b.get('type')=='tool_result' and not b.get('is_error'):success.add(b.get('tool_use_id'))
                        if b.get('type')!='tool_use':continue
                        inp=b.get('input',{});
                        if not isinstance(inp,dict):continue
                        path=Path(inp.get('file_path',''))
                        if not path.is_absolute():path=run/'workspace'/path
                        if b.get('name')=='Read':
                            if not path.resolve().is_relative_to((run/'workspace').resolve()):row['external_file_reads'].append(str(path))
                            if path.name=='simulator-evidence.json':read_ids[b['id']]=order
                        if b.get('name')=='Edit' and path.name=='candidate.py':
                            row['candidate_edits']+=1
                            if first_edit is None:first_edit=order
                        if b.get('name')=='Bash' and ' decision ' in inp.get('command',''):
                            if decision_request is None:decision_request=order
                row['simulator_reads']=[dict(tool_id=k,event_order=v,returned_without_tool_error=k in success) for k,v in read_ids.items()]
                row['decision_requested_before_first_edit']=first_edit is None or (decision_request is not None and decision_request<first_edit)
            for dec in (run/'trace/decisions').glob('*.json'):
                d=json.loads(dec.read_bytes())
                if d['round_id']==call.name:row['decisions'].append(dict(path=str(dec.relative_to(r)),sha256=sha(dec),record=d))
            if output.exists():
                patch='\n'.join(difflib.unified_diff(source.read_text().splitlines(),output.read_text().splitlines(),fromfile='parent',tofile='candidate'))+'\n'
                (call/'candidate.diff').write_text(patch)
                formal=run/('formal-'+sha(output)[:12])/'evaluation/result.json'
                if formal.exists():
                    f=json.loads(formal.read_bytes());row['formal']={k:f.get(k) for k in ['status','num_passed','num_workloads','geo_mean']}
                    row['timings']=[{k:x.get(k) for k in ['uuid','status','latency_ms','reference_latency_ms','speedup']} for x in f['per_workload'] if x['phase']=='timing']
                pre=run/'attempts'/sha(output)/'preflight.json'
                if pre.exists():row['preflight']=json.loads(pre.read_bytes())
            rows.append(row)
        result['arms'][arm]=rows
    (r/'trace-inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:len(v) for k,v in result['arms'].items()}))


if __name__=='__main__':main()
