"""Continue an authorized fixed-source round only after its actual artifacts finish."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True)
    p.add_argument('--arm',choices=['no-simulator','with-simulator'],required=True)
    p.add_argument('--completed-round',required=True);p.add_argument('--next-round',required=True)
    p.add_argument('--source-sha',required=True);p.add_argument('--paired',type=Path,required=True)
    p.add_argument('--simulator',type=Path);a=p.parse_args();r=a.study;run=r/a.arm
    budget=json.loads((r/'experiment.json').read_bytes())['budget_per_arm']['generation_calls']
    assert int(a.next_round.split('-')[-1])<=budget
    assert (a.arm=='with-simulator')==bool(a.simulator)
    folder=r/'continuations';folder.mkdir(exist_ok=True)
    record=folder/(a.arm+'-'+a.next_round+'.json')
    state={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    state.update(status='waiting',started_at=time.time())
    with record.open('x') as f:json.dump(state,f,indent=2)
    try:
        deadline=time.monotonic()+3600
        required=[run/'assessments'/a.completed_round/'result.json',a.paired/'comparison.json']
        if a.simulator:required.append(a.simulator/'execution.json')
        while True:
            assert hashlib.sha256((run/'workspace/candidate.py').read_bytes()).hexdigest()==a.source_sha,'source changed while awaiting artifacts'
            assert not (run/('process-'+a.next_round+'.json')).exists(),'next call already launched'
            if all(x.exists() for x in required):
                try:
                    documents=[json.loads(x.read_bytes()) for x in required]
                except json.JSONDecodeError:
                    time.sleep(2);continue
                assert documents[0]['source_sha256']==a.source_sha
                assert json.loads((a.paired/'spec.json').read_bytes())['source_sha256']==a.source_sha
                preflight=documents[0].get('preflight',{}).get('status')
                assert preflight in ['passed','failed'],'assessment has no terminal preflight result'
                # Numerical failures are feedback too; no formal timing is valid or
                # required for them. Successful preflight still requires formal results.
                formal=run/('formal-'+a.source_sha[:12])/'evaluation/result.json'
                if preflight=='passed' and not formal.exists():
                    time.sleep(5);continue
                break
            if time.monotonic()>deadline:raise TimeoutError('Artifacts not ready; no new model call launched')
            time.sleep(5)
        command=[sys.executable,str(r/'contrast_next_call.py'),a.arm,a.next_round,'--study',str(r),'--paired',str(a.paired)]
        if a.simulator:
            subprocess.run([sys.executable,str(r/'contrast_simulator_summary.py'),'--study',str(r),'--directory',str(a.simulator)],check=True)
            summary=a.simulator/'summary.json';s=json.loads(summary.read_bytes())
            assert s['source_sha256']==a.source_sha and s['status'] in ['qualified_observation','completed'] and s.get('reports')
            command+=['--simulator',str(summary)]
        subprocess.run(command,check=True)
        state['status']='next_model_call_started'
    except BaseException as error:
        state.update(status='stopped_without_continuation',error=repr(error));raise
    finally:
        state['finished_at']=time.time();record.write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':main()
