"""Return actual preceding results and source-bound artifacts for a subsequent call."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('arm');p.add_argument('round');p.add_argument('--study',type=Path,required=True);p.add_argument('--simulator',type=Path);p.add_argument('--paired',type=Path);a=p.parse_args()
    r=a.study;run=r/a.arm;w=run/'workspace';source=w/'candidate.py'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    save=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    digest=sha(source);formal=run/('formal-'+digest[:12]);attempt=run/'attempts'/digest
    feedback={'source_sha256':digest,'formal_available':False}
    if (formal/'evaluation/result.json').exists():
        result=json.loads((formal/'evaluation/result.json').read_bytes())
        feedback.update(formal_available=True,status=result['status'],num_passed=result['num_passed'],num_workloads=result['num_workloads'],geo_mean=result.get('geo_mean'),
           rows=[{k:row.get(k) for k in ['uuid','phase','status','latency_ms','reference_latency_ms','speedup','log']} for row in result['per_workload']])
        for row in feedback['rows']:
            if isinstance(row.get('log'),str):row['log']=row['log'][-2500:]
    elif (attempt/'preflight.json').exists():feedback['preflight']=json.loads((attempt/'preflight.json').read_bytes())
    else:raise ValueError('No actual feedback for current source')
    save(w/'previous-result.json',feedback)
    artifacts=[]
    verification=attempt/'compiled/verification.json'
    if verification.exists():
        meta=json.loads(verification.read_bytes());assert meta['source_sha256']==digest
        folder=w/'current-ir'/digest[:12];folder.mkdir(parents=True,exist_ok=True)
        for i,entry in enumerate(meta['entries']):
            for item in entry['artifacts']:
                if item['kind'] not in ['ttir','bcmlir']:continue
                path=Path(item['path']);assert sha(path)==item['sha256']
                dest=folder/(str(i)+'-'+path.name);shutil.copyfile(path,dest)
                artifacts.append({'path':str(dest.relative_to(w)),'sha256':sha(dest)})
    c=json.loads((run/'config.json').read_bytes())
    current_profiling=None
    if a.paired:
        binding=json.loads((a.paired/'spec.json').read_bytes());assert binding['source_sha256']==digest
        comparison=json.loads((a.paired/'comparison.json').read_bytes());facts=comparison.get('result',comparison)
        save(w/'current-profiling.json',dict(source_sha256=digest,case=binding['case'],
            candidate_basic=facts['candidate'].get('basic'),reference_basic=facts['reference'].get('basic'),
            differences=facts['differences'],derived_differences=facts['derived_differences'],scope=facts['scope'],limitations=facts['limitations']))
        current_profiling='current-profiling.json'
    if a.arm=='with-simulator':
        assert a.simulator
        sim=json.loads(a.simulator.read_bytes());assert sim['source_sha256']==digest
        shutil.copyfile(a.simulator,w/'simulator-evidence.json')
        c['required_simulator_evidence']={'path':'simulator-evidence.json','sha256':sha(w/'simulator-evidence.json')}
    save(w/'context-brief.json',dict(source_sha256=digest,current_result='previous-result.json',actual_ir=artifacts,
        historical_profiling='profiling-comparison.json is original-source evidence only; use as background unless its binding matches current source.',
        current_profiling=current_profiling,current_simulator='simulator-evidence.json' if a.arm=='with-simulator' else None,
        instruction='Read actual feedback; do not reuse an old source score or diagnose current source using old metrics as if current.'))
    save(run/'config.json',c)
    prompt=(r/'comparison-prompt.md').read_text()+'\nThis is '+a.round+'. Read previous-result.json and current context-brief.json first, including current-profiling.json when listed. Previous formal failure is part of the trace, not a measured speedup. Repair a reported interface/compile/correctness problem before claiming performance. If current IR exists, its paths are in context-brief.json. Original profiling and original runtime metadata remain historical background; do not assert they describe the current candidate. Register the current source decision before editing.\n'
    prompt_path=run/(a.round+'-prompt.md');prompt_path.write_text(prompt)
    cmd=[sys.executable,'-m','hgkda.runtime.agent_runner','--config',str(run/'config.json'),'--model-settings',str(run/'model-settings.json'),'--invocation',a.round,'--max-turns','48','--prompt-file',str(prompt_path)]
    with (run/('generation-'+a.round+'.log')).open('x') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH=str(r/'method/hgkda/src')),start_new_session=True)
    save(run/('process-'+a.round+'.json'),dict(pid=proc.pid,command=cmd))
    print(json.dumps(dict(arm=a.arm,round=a.round,pid=proc.pid,source_sha256=digest)))


if __name__=='__main__':main()
