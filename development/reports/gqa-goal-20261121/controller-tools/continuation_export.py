"""Export small immutable evidence and close only this continuation's idle service."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tarfile
import time
import urllib.request
import urllib.error

r=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/data/hanle/ascend-optimization/runs/upsample-performance-20261118')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
save=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
selection=json.loads((r/'best/selection.json').read_bytes());digest=selection['source_sha256']
assert sha(r/'best/candidate.py')==digest
service=json.loads((r/'service.json').read_bytes())
server_url=service['server'];port=server_url.rsplit(':',1)[1].encode()
try:
    with urllib.request.urlopen(server_url+'/status',timeout=15) as response:status=json.load(response)
except urllib.error.URLError:
    assert (r/'service-closeout.json').exists()
    status=json.loads((r/'service-final-status.json').read_bytes())
assert status['scheduler']['active']==0 and status['scheduler']['waiting']==0
save(r/'service-final-status.json',status)
# Identify the exact container-side server by its project-specific argv path and port.
owned=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:args=(proc/'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    if b'kernelgen_server.server' in args and port in args and str(r/'server-profiles').encode() in args:
        owned.append(int(proc.name))
assert len(owned)==1 or (not owned and (r/'service-closeout.json').exists()),owned
for pid in owned:os.kill(pid,signal.SIGTERM)
if owned:
    for _ in range(30):
        if all(not (Path('/proc')/str(pid)).exists() for pid in owned):break
        time.sleep(1)
    assert all(not (Path('/proc')/str(pid)).exists() for pid in owned)
    save(r/'service-closeout.json',dict(status='idle service stopped',server_pids=owned,
        launcher_pid=service['pid'],basis='Exact argv match, owned profile path and port; scheduler idle'))
else:
    closed=json.loads((r/'service-closeout.json').read_bytes());closed['status']='idle service stopped'
    save(r/'service-closeout.json',closed)
files=[]
for p in r.rglob('*'):
    if not p.is_file():continue
    parts=p.relative_to(r).parts
    if any(x in parts for x in ['method','method-package','server-cache','server-profiles','server-bundles','server-debug',
        'triton-cache','profile-triton-cache','checker-cache','compiler-cache','raw','export_only_prof_dir',
        '__pycache__','skills']):continue
    if p.name in ['model-settings.json','evidence.tar.gz','evidence-manifest.json']:continue
    if p.suffix in ['.so','.o','.npubin']:continue
    files.append(p)
manifest=dict(source_sha256=digest,files=[dict(path=str(p.relative_to(r)),sha256=sha(p),bytes=p.stat().st_size) for p in files],
    excluded='Caches, raw simulator dumps, binary objects, model settings and service working directories stay remote.')
save(r/'evidence-manifest.json',manifest)
with tarfile.open(r/'evidence.tar.gz','w:gz') as tar:
    for p in files:tar.add(p,arcname=str(p.relative_to(r)),recursive=False)
    tar.add(r/'evidence-manifest.json',arcname='evidence-manifest.json')
print(json.dumps(dict(files=len(files),archive_sha256=sha(r/'evidence.tar.gz'),bytes=(r/'evidence.tar.gz').stat().st_size)))
