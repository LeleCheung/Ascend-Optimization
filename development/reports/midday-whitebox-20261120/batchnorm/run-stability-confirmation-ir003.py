from pathlib import Path
import json,subprocess,datetime,hashlib
r=Path('/data/hanle/ascend-optimization/runs/midday-whitebox-20261120/batchnorm')
py='/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python'
tool=r.parent/'tools/ir_stability.py'
plan=json.loads((r/'stability-confirmation-ir-003.json').read_bytes())
records=[]
for rel in plan['outputs']:
 out=r/rel
 assert not out.exists()
 event=dict(output=rel,started_at=datetime.datetime.utcnow().isoformat())
 telemetry=[]
 for kind in ['usages','sensors','proc-mem']:
  p=subprocess.run(['docker','exec','tle_yy','npu-smi','info','-t',kind,'-i','7','-c','0'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  telemetry.append(dict(kind=kind,returncode=p.returncode,text=p.stdout.decode()))
 event['before_device_snapshot']=telemetry
 command=[py,str(tool),'--run',str(r),'--invocation','ir-003','--kgs-root','/data/hanle/ascend-optimization/runtime/repository-migration-20261111/external-backends/kernelgen_server','--output',str(out)]
 with (r/(Path(rel).name+'.log')).open('x') as log:
  p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 event.update(returncode=p.returncode,finished_at=datetime.datetime.utcnow().isoformat(),command=command)
 if p.returncode==0:
  summary=out/'summary.json'
  event.update(summary=json.loads(summary.read_bytes()),summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest())
 records.append(event)
 (r/'stability-confirmation-results-ir003.json').write_text(json.dumps(dict(status='confirmations_running' if len(records)<len(plan['outputs']) else 'confirmations_complete',records=records,automatic_retention=False),indent=2)+chr(10))
 if p.returncode:break
