"""Wait for final assessment, then replay the best eligible unchanged source once."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True)
    p.add_argument('--arm',choices=['no-simulator','with-simulator'],required=True)
    a=p.parse_args();r=a.study;run=r/a.arm
    record=r/'continuations'/(a.arm+'-final-replay.json')
    state=dict(status='waiting_for_final_assessment',arm=a.arm,started_at=time.time())
    with record.open('x') as f:json.dump(state,f,indent=2)
    try:
        deadline=time.monotonic()+3600
        while not (run/'assessments/round-08/result.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('final assessment not finished')
            time.sleep(5)
        completed=list((run/'agent-calls').glob('round-*/output-source.py'))
        assert len(completed)==8,'generation budget not finished'
        subprocess.run([sys.executable,str(r/'contrast_retention.py'),'--study',str(r)],check=True)
        snapshot=sorted((r/'retention').glob('*.json'))[-1]
        selection=json.loads(snapshot.read_bytes())['arms'][a.arm]['best_full_contract']
        assert selection and selection['eligible']
        state.update(status='replaying',selection=selection,retention_snapshot=str(snapshot))
        record.write_text(json.dumps(state,indent=2))
        subprocess.run([sys.executable,str(r/'contrast_final_replay.py'),'--study',str(r),
                        '--arm',a.arm,'--source-sha',selection['source_sha256']],check=True)
        state['status']='completed'
    except BaseException as e:
        state.update(status='stopped',error=repr(e));raise
    finally:
        state['finished_at']=time.time();record.write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':main()
