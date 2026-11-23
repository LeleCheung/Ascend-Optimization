"""Read-only inventory for the requested machine-specific indicator review."""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'development/reports/machine-indicator-audit-20261121'


def remote(alias: str, code: str, name: str) -> dict:
    encoded = base64.b64encode(code.encode()).decode()
    expression = f'import base64;exec(base64.b64decode({encoded!r}))'
    command = shlex.join(['python3', '-c', expression])
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o',
                             'ConnectTimeout=12', alias, command],
                            capture_output=True, timeout=55)
    output = result.stdout.decode('utf-8', errors='replace')
    record = {'alias': alias, 'returncode': result.returncode,
              'stdout': output, 'stderr': result.stderr.decode('utf-8', errors='replace')}
    (OUT / name).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    return record


INVENTORY = r'''
import json,subprocess,shutil,pathlib
data={}
def run(args):
 try:
  r=subprocess.run(args,capture_output=True,text=True,timeout=15)
  text=r.stdout
  if args[0]=='lspci':text='\n'.join(x for x in text.splitlines() if any(k in x.lower() for k in ['vga','display','3d controller','processing accelerators','moore','ascend']))
  return {'command':args,'returncode':r.returncode,'stdout':text[:14000],'stderr':r.stderr[:1500]}
 except Exception as e:return {'command':args,'error':str(e)}
data['architecture']=run(['uname','-m'])
for tool in ['npu-smi','mthreads-gmi','mtgmi','mcc','mcu','msys','msprof','docker','lspci']:
 p=shutil.which(tool)
 if p:data.setdefault('executables',{})[tool]=p
for args in [['npu-smi','info'],['mthreads-gmi','-L'],['docker','ps','--format','{{.Names}}\t{{.Image}}'],['lspci','-nn']]:
 if shutil.which(args[0]):data[args[0]+' '+args[1]]=run(args)
for root in ['/opt','/usr/local']:
 p=pathlib.Path(root)
 data[root]=[x.name for x in p.iterdir() if any(k in x.name.lower() for k in ['musa','moore','perf','ascend','conda','python'])]
print(json.dumps(data,ensure_ascii=False,indent=2))
'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[('baai-910b',INVENTORY,'ascend-host-inventory.json'),
              ('baai-s5000',INVENTORY,'mthreads-host-inventory.json')]
        for result in pool.map(lambda args: remote(*args),jobs):
            print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
