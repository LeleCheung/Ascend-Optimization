"""Start a recorded model hypothesis once the exact starting evidence is ready."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--run',type=Path,required=True)
p.add_argument('--controller-pid',type=int,required=True)
p.add_argument('--invocation',required=True)
a=p.parse_args();r=a.run
status=r/'starting-evidence-status.json'
while True:
    value=json.loads(status.read_bytes()) if status.exists() else {}
    if value.get('status')=='evidence_ready':break
    proc=Path('/proc')/str(a.controller_pid)
    if not proc.exists():
        raise RuntimeError('Evidence controller exited; preserve logs and repair collection before generation')
    command=(proc/'cmdline').read_bytes()
    if str(r).encode() not in command or b'retained_research_evidence.py' not in command:
        raise RuntimeError('Evidence controller identity changed')
    time.sleep(5)
assert not (r/'agent-calls'/a.invocation).exists()
subprocess.run([sys.executable,str(r.parent/'continuation_next.py'),
    '--run',str(r),'--invocation',a.invocation],check=True)
