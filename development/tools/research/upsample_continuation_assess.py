"""Validate one reviewed continuation candidate and retain qualified improvements."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--invocation',required=True)
    a=p.parse_args();r=a.run;w=r/'workspace';source=w/'candidate.py'
    read=lambda path:json.loads(path.read_bytes())
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    digest=sha(source);review=r/'source-reviews'/(digest+'.json')
    from hgkda.gate import admit
    admit(source.read_text(encoding='utf8'),read(review))
    assert sha(r/'agent-calls'/a.invocation/'output-source.py')==digest
    dest=r/'assessments'/a.invocation;dest.mkdir(parents=True,exist_ok=False)
    config=read(r/'config.json');config['trace_round_id']=a.invocation
    save(dest/'config.json',config)
    env=dict(os.environ,PYTHONPATH=str(r/'method/hgkda/src'))
    result=dict(source_sha256=digest,invocation=a.invocation,started_at=datetime.now(timezone.utc).isoformat())
    def execute(name,args):
        with (dest/(name+'.log')).open('x') as log:
            code=subprocess.run([sys.executable,*args],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        result[name+'_returncode']=code;save(dest/'result.json',result);return code
    rc=execute('preflight',['-m','hgkda.runtime.agent_tools','--config',str(dest/'config.json'),'preflight','--source',str(source)])
    pre=w/'latest-preflight.json'
    result['preflight']=read(pre) if pre.exists() else {}
    if rc==0 and result['preflight'].get('status')=='passed':
        execute('formal',['-m','hgkda.runtime.evaluate','--config',str(dest/'config.json'),'--source',str(source),'--review',str(review)])
    formal=r/('formal-'+digest[:12]);audit=formal/'audit-final.json'
    result['qualified']=False
    if audit.exists():
        row=read(audit)['rows'][0]
        result['qualified']=row['source_sha256']==digest and row['full_passed'] and all(x['status']=='passed' for x in row['checks'].values())
    state=read(r/'continuation.json')
    event=dict(invocation=a.invocation,source_sha256=digest,qualified=result['qualified'],assessment=str(dest))
    if result['qualified']:
        measured=read(formal/'evaluation/result.json')
        timings=[x for x in measured['per_workload'] if x['phase']=='timing']
        target_id=config['experiment_goal']['target_case']
        target=next(x for x in timings if x['uuid']==target_id)
        event.update(display_round=state.get('prior_display_rounds',3)+1+len(state['qualified_rounds']),target_latency_ms=target['latency_ms'],
                     target_speedup=target['speedup'],geo_mean=measured['geo_mean'],timings=timings)
        selection=read(r/'best/selection.json')
        if 'target_latency_ms' in selection:best_latency=selection['target_latency_ms']
        else:
            original=next((r/'best/final-replay').glob('formal-*/evaluation/result.json'))
            best_latency=next(x['latency_ms'] for x in read(original)['per_workload'] if x['uuid']==target['uuid'])
        event['retained']=target['latency_ms']<best_latency
        event['previous_best_latency_ms']=best_latency
        if event['retained']:
            # Copy validated bytes, then publish their identity; exploration source stays separate.
            shutil.copyfile(formal/'candidate.py',r/'best/candidate.py')
            save(r/'best/selection.json',dict(source_sha256=digest,origin=str(formal/'candidate.py'),
                display_round=event['display_round'],target_latency_ms=target['latency_ms'],
                target_speedup=target['speedup'],geo_mean=measured['geo_mean'],validation=str(formal)))
            state['best_source_sha256']=digest
        state['qualified_rounds'].append(event)
    state['attempts'].append(event);state['status']='assessment_complete'
    result['event']=event;save(dest/'result.json',result);save(r/'continuation.json',state)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
