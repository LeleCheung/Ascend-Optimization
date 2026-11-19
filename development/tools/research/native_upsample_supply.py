"""Supply identical non-simulator evidence and start one explicit study call."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

r=Path('/data/hanle/ascend-optimization/runs/native-upsample-contrast-20261118')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['no-simulator','with-simulator']);p.add_argument('--start',action='store_true');a=p.parse_args()
    run=r/a.arm;w=run/'workspace';source=w/'candidate.py';digest=sha(source)
    assert digest==sha(r/'upstream.py')
    result=json.loads((r/'native-result.json').read_bytes())
    summary={'source_sha256':digest,'status':result['status'],'num_passed':result['num_passed'],'num_workloads':result['num_workloads'],'geo_mean':result.get('geo_mean'),
             'timing_mode':'KGS profiler; milliseconds','rows':[{k:row.get(k) for k in ['uuid','phase','status','latency_ms','reference_latency_ms','speedup','error','log']} for row in result['per_workload']]}
    for row in summary['rows']:
        for field in ['log','error']:
            if isinstance(row.get(field),str):row[field]=row[field][:3500]
    save(w/'baseline-evaluation.json',summary)
    shutil.copyfile(r/'native-runtime-facts.json',w/'native-runtime-facts.json')
    pair=r/'native/paired'
    shutil.copyfile(pair/'comparison.json',w/'profiling-comparison.json')
    ir=w/'actual-ir';ir.mkdir(exist_ok=True)
    for item in (pair/'actual-ir').iterdir():
        if item.suffix in ['.ttir','.bcmlir'] or item.name=='manifest.json':shutil.copyfile(item,ir/item.name)
    files=['candidate.py','contract.json','knowledge.json','baseline-evaluation.json','native-runtime-facts.json','profiling-comparison.json']+[str(x.relative_to(w)) for x in ir.iterdir()]
    context={'source_sha256':digest,'origin':'untouched fixed FlagGems','target_case':'upsample_bilinear2d-timing-1','formal_contract_cases':24,'full_original_result':'18/24; 6 compilation failures; all 6 timing cases ran',
             'prior_optimization_answers_provided':False,'compatibility_phase':'A separate preparation candidate was not used as a starting point; both arms start original bytes.',
             'evidence':files,'timing':'profiler','simulator_provided':a.arm=='with-simulator'}
    c=json.loads((run/'config.json').read_bytes())
    if a.arm=='with-simulator':
        sim=r/'native/selected-replay-v2/summary.json';s=json.loads(sim.read_bytes());assert s['source_sha256']==digest
        shutil.copyfile(sim,w/'simulator-evidence.json');files.append('simulator-evidence.json')
        c['required_simulator_evidence']={'path':'simulator-evidence.json','sha256':sha(w/'simulator-evidence.json')}
    save(w/'context-brief.json',context)
    save(w/'evidence-index.json',{'source_sha256':digest,'files':[dict(path='workspace/'+name,sha256=sha(w/name)) for name in files]})
    save(run/'config.json',c)
    if not a.start:return
    cmd=[sys.executable,'-m','hgkda.runtime.agent_runner','--config',str(run/'config.json'),'--model-settings',str(run/'model-settings.json'),'--invocation','round-01','--max-turns','48','--prompt-file',str(r/'comparison-prompt.md')]
    with (run/'generation-round-01.log').open('x') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH=str(r/'method/hgkda/src')),start_new_session=True)
    save(run/'process-round-01.json',{'pid':child.pid,'command':cmd})
    print(json.dumps({'arm':a.arm,'pid':child.pid,'source_sha256':digest}))

if __name__=='__main__':main()
