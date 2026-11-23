"""Launch a recorded continuation call and a project-owned KGS service."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

r=Path('/data/hanle/ascend-optimization/runs/upsample-performance-20261118')
old=r.parent/'native-upsample-contrast-20261118'
save=lambda p,v:p.write_text(json.dumps(v,indent=2)+'\n')
if not (r/'controller-state.json').exists():save(r/'controller-state.json',dict(status='prepared'))
# Record the failed pre-model setup separately; the original invocation remains intact.
invocation=sys.argv[1]
assert not (r/'agent-calls'/invocation).exists()
source=r/'workspace/candidate.py'
import hashlib
best=json.loads((r/'best/selection.json').read_bytes())
assert hashlib.sha256(source.read_bytes()).hexdigest()==best['source_sha256'],'new hypothesis starts from retained best'
env=dict(os.environ,PYTHONPATH=str(r/'method/hgkda/src'))
cmd=[sys.executable,'-m','hgkda.runtime.agent_runner','--config',str(r/'config.json'),
     '--model-settings',str(r/'model-settings.json'),'--invocation',invocation,'--prompt-file',str(r/'prompt.md')]
with (r/f'generation-{invocation}.log').open('x') as log:
    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,cwd=r,start_new_session=True)
save(r/f'process-{invocation}.json',dict(pid=proc.pid,command=cmd,starting_source_sha256=best['source_sha256']))
if not (r/'service.json').exists():
    with socket.socket() as check:
        assert check.connect_ex(('127.0.0.1',19728))!=0,'KGS port already occupied'
    service=json.loads((old/'service.json').read_bytes())
    command=[part.replace(str(old),str(r)) for part in service['command']]
    with (r/'kgs.log').open('x') as log:
        server=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    save(r/'service.json',dict(pid=server.pid,command=command,server='http://127.0.0.1:19728',timing='profiler',scope='project-owned'))
print(json.dumps(dict(invocation=invocation,pid=proc.pid)))
