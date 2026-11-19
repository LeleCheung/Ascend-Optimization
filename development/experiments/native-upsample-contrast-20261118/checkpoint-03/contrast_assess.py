"""Assess a specifically reviewed generated source once and preserve all failures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('arm');p.add_argument('round');p.add_argument('sha');a=p.parse_args()
    run=a.study/a.arm;source=run/'workspace/candidate.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==a.sha
    review=run/'source-reviews'/(a.sha+'.json');assert review.exists()
    out=run/'assessments'/a.round;out.mkdir(parents=True,exist_ok=False)
    c=json.loads((run/'config.json').read_bytes());c['trace_round_id']=a.round
    config=out/'config.json';config.write_text(json.dumps(c,indent=2))
    env=dict(os.environ,PYTHONPATH=str(a.study/'method/hgkda/src'))
    start=time.time();results={'source_sha256':a.sha,'round':a.round}
    def execute(name,args):
        with (out/(name+'.log')).open('x') as log:
            rc=subprocess.run([sys.executable,*args],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        results[name+'_returncode']=rc
        return rc
    rc=execute('preflight',['-m','hgkda.runtime.agent_tools','--config',str(config),'preflight','--source',str(source)])
    pre=run/'workspace/latest-preflight.json'
    if pre.exists():results['preflight']=json.loads(pre.read_bytes())
    if rc==0 and results.get('preflight',{}).get('status')=='passed':
        execute('formal',['-m','hgkda.runtime.evaluate','--config',str(config),'--source',str(source),'--review',str(review)])
    results['elapsed_seconds']=time.time()-start
    (out/'result.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results))


if __name__=='__main__':main()
