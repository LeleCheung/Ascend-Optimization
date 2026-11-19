"""Replay an explicitly selected unchanged candidate under the frozen full contract."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--arm',choices=['no-simulator','with-simulator'],required=True)
    p.add_argument('--source-sha',required=True)
    a=p.parse_args();r=a.study;run=r/a.arm
    source=next(x for x in (run/'agent-calls').glob('*/output-source.py')
                if hashlib.sha256(x.read_bytes()).hexdigest()==a.source_sha)
    formal=run/('formal-'+a.source_sha[:12])
    old=json.loads((formal/'evaluation/result.json').read_bytes())
    assert old['status']=='PASSED' and old['num_passed']==24
    for name in ['supplementary.json','cann_supplementary.json']:
        check=json.loads((formal/name).read_bytes())
        assert check['status']=='passed' and check['source_sha256']==a.source_sha
    out=r/'final-replay'/a.arm;out.mkdir(parents=True,exist_ok=False)
    (out/'workspace').mkdir()
    config=json.loads((run/'config.json').read_bytes())
    config['run_directory']=str(out)
    config['trace_round_id']='final-replay'
    config_path=out/'config.json';config_path.write_text(json.dumps(config,indent=2))
    record=dict(source_sha256=a.source_sha,source=str(source),original_result=str(formal/'evaluation/result.json'),
                reason='Final paired comparison: small-case CANN reference drift up to 10%, larger than intermediate aggregate arm gap; one unchanged-source full replay per arm',
                scope='Post-search validation; no new generation session or candidate edit')
    (out/'selection.json').write_text(json.dumps(record,indent=2))
    env=dict(os.environ,PYTHONPATH=str(r/'method/hgkda/src'))
    command=[sys.executable,'-m','hgkda.runtime.evaluate','--config',str(config_path),
             '--source',str(source),'--review',str(run/'source-reviews'/(a.source_sha+'.json'))]
    with (out/'evaluation.log').open('x') as log:
        rc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    (out/'execution.json').write_text(json.dumps(dict(returncode=rc,source_sha256=a.source_sha)))
    print(json.dumps(dict(arm=a.arm,returncode=rc,directory=str(out))))
    sys.exit(rc)


if __name__=='__main__':main()
