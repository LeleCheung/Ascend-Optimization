"""Collect public documentation and read-only device evidence for the matrix."""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'development/reports/ascend-statistics-20261120'


def remote(code: str, filename: str) -> None:
    encoded = base64.b64encode(code.encode()).decode()
    expression = f'import base64;exec(base64.b64decode({encoded!r}))'
    command = shlex.join(['docker', 'exec', 'tle_yy', '/usr/local/python3.11.15/bin/python3.11', '-c', expression])
    result = subprocess.run(['ssh', 'baai-910b', command], capture_output=True, timeout=150)
    (OUT / filename).write_bytes(result.stdout)
    if result.stderr or result.returncode:
        (OUT / (filename + '.stderr')).write_bytes(result.stderr)
    print(json.dumps({'file': filename, 'bytes': len(result.stdout), 'returncode': result.returncode}))


def public(repo: str, commit: str, path: str) -> dict:
    url = f'https://raw.githubusercontent.com/Ascend/{repo}/{commit}/{path}'
    data = urllib.request.urlopen(url, timeout=45).read()
    name = repo + '-' + Path(path).name
    (OUT / 'official' / name).write_bytes(data)
    return dict(url=url, commit=commit, local='official/' + name, bytes=len(data))


def main() -> None:
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / 'official').mkdir(exist_ok=True)
    jobs = [
        ('msopprof', 'a74e105fcfad06c3faa4be308834b3158e221299', 'docs/zh/user_guide/' + p)
        for p in ['msopprof_performance_data.md', 'msopprof_simulator_performance_data.md', 'msopprof_simulator_user_guide.md']
    ] + [('msprof', 'a49240a2086ea472cb12a4a2399b395ebd5fd8fc', 'docs/zh/user_guide/profile_data_file_references.md')]
    with ThreadPoolExecutor(max_workers=4) as pool:
        sources = list(pool.map(lambda args: public(*args), jobs))
    (OUT / 'official-sources.json').write_text(json.dumps(sources, indent=2), encoding='utf-8')
    remote('''import json, subprocess, pathlib
data={}
for name,args in {
 'board':['npu-smi','info','-t','board','-i','7'],
 'memory':['npu-smi','info','-t','memory','-i','7'],
 'common':['npu-smi','info','-t','common','-i','7'],
 'product':['npu-smi','info','-t','product','-i','7']}.items():
 r=subprocess.run(args,capture_output=True,text=True)
 lines=[x for x in r.stdout.splitlines() if not any(k in x.lower() for k in ['serial','board id','barcode'])]
 data[name]=dict(command=args,returncode=r.returncode,stdout='\\n'.join(lines),stderr=r.stderr)
root=pathlib.Path('/usr/local/Ascend/cann-9.0.0')
hits=list(root.rglob('*910B*.ini'))+list(root.rglob('*910b*.ini'))+list(root.rglob('platform_ascendc.h'))
data['sdk_files']=[str(x) for x in hits]
data['sdk_configs']={str(x):x.read_text(errors='replace') for x in hits if x.suffix=='.ini'}
print(json.dumps(data,indent=2))
''', 'device-query.json')
    remote('''import json,pathlib
base=pathlib.Path('/data/hanle/ascend-optimization/runs/night-whitebox-20261120')
data={}
for op in ['native-group-norm','gqa']:
 d=json.loads((base/op/'paired-parent/comparison.json').read_text())
 data[op]=d
print(json.dumps(data,indent=2))
''', 'paired-observations.json')


if __name__ == '__main__':
    main()
