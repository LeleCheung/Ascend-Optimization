"""Capture a completed qualified hypothesis while preserving failed assessments."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--invocation',required=True)
p.add_argument('--assessment-pid',type=int,required=True);p.add_argument('--public-package',type=Path,required=True)
a=p.parse_args();r=a.run;result=r/'assessments'/a.invocation/'result.json'
while True:
    record=json.loads(result.read_bytes()) if result.exists() else {}
    if 'qualified' in record:break
    if not Path('/proc/'+str(a.assessment_pid)).exists():
        raise RuntimeError('Assessment process exited before publishing its result')
    time.sleep(5)
if not record['qualified']:
    print(json.dumps(dict(status='assessment_requires_repair',source_sha256=record['source_sha256'])))
    sys.exit(0)
digest=record['source_sha256'];sim=r/('simulator-'+digest[:8])
capture=subprocess.run([sys.executable,str(r.parent/'upsample_continuation_capture.py'),digest,str(r)])
if capture.returncode:raise RuntimeError('Capture controller requires inspection')
if not (sim/'summary.json').exists():
    subprocess.run([sys.executable,str(r.parent/'summarize_instruction_observation.py'),'--run',str(r),'--directory',str(sim)],check=True)
subprocess.run([sys.executable,str(r.parent/'relative_timeline_report.py'),'--directory',str(sim),'--package',str(a.public_package)],check=True)
print(json.dumps(dict(status='postchange_evidence_ready',source_sha256=digest)))
