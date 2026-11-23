"""Prepare a raw native QKV baseline and isolated source-bound research evidence."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

r=Path('/data/hanle/ascend-optimization/runs/qkv-simulator-20261118')
root=r.parent.parent
repo=root/'Ascend-Optimization'
prior=root/'runs/upsample-performance-20261118'
env=json.loads((prior/'config.json').read_bytes())
environment={key:env[key] for key in ['container','device','device_python','sdk','lock','kgs_server']}
environment['kgs_server']='http://127.0.0.1:19729'
environment_file=root/'runs/qkv-environment-20261118.json'
environment_file.write_text(json.dumps(environment,indent=2))
subprocess.run([sys.executable,'-m','hgkda.runtime.prepare','--operator-directory',str(repo/'operators/transform_bias_rescale_qkv'),
    '--package',str(root/'runtime/hgkda-performance-first-20261118/hgkda'),'--environment',str(environment_file),'--run-directory',str(r)],check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
save=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
source=repo/'development/dependencies/third_party/FlagGems/src/flag_gems/ops/_transform_bias_rescale_qkv.py'
native=r/'native';native.mkdir()
for p in [native/'original.py',r/'workspace/candidate.py',r/'workspace/starter.py']:shutil.copyfile(source,p)
save(r/'workspace/starter-provenance.json',dict(kind='fixed_native_flaggems',source_sha256=sha(source),origin=str(source),prior_best_source_provided=False))
c=json.loads((r/'config.json').read_bytes())
c.update(context_kind='research_native',goal_kind='performance_recovery_with_formal_feedback',
    capture_entrypoint='_transform_bias_rescale_qkv',capture_warmup=0,
    device_pythonpath=str(repo/'development/dependencies/third_party/FlagGems/src'),
    maximum_preflight_sources=0,maximum_formal_sources=0,optimization_policy='performance_first',
    public_entrypoint='run(qkv,qkv_bias,num_heads)',experiment_goal=dict(kind='simulator_guided_optimization',
    target_case='transform_bias_rescale_qkv-timing-2',new_control_arm=False))
save(r/'config.json',c)
request=json.loads((r/'request.json').read_bytes())
request['implementation']['sources']=[dict(path='main.py',content=source.read_text())]
request['implementation']['entrypoint']='main.py::_transform_bias_rescale_qkv'
request['implementation']['name']='native_qkv_20261118'
save(native/'request.json',request)
shutil.copyfile(prior/'model-settings.json',r/'model-settings.json')
for name in ['regions_optimized.py','contrast_simulator_summary.py']:
    shutil.copyfile(prior/name,r/name)
save(r/'study.json',dict(operator=c['operator'],status='native_collection',source_sha256=sha(source),
    source=str(native/'original.py'),target_case=c['experiment_goal']['target_case'],
    provenance='Frozen upstream module; reference entrypoint selected by request metadata, source bytes unchanged.',
    old_delivery_scope='Historical complete-contract score; separate from new profiler target-case trace.'))
with socket.socket() as check:assert check.connect_ex(('127.0.0.1',19729))!=0
service=json.loads((prior/'service.json').read_bytes())
cmd=[x.replace(str(prior),str(r)).replace('19728','19729') for x in service['command']]
with (r/'kgs.log').open('x') as log:
    server=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
save(r/'service.json',dict(pid=server.pid,command=cmd,server=c['kgs_server'],scope='project-owned',timing='profiler'))
print(json.dumps(dict(run=str(r),source_sha256=sha(source),service_pid=server.pid)))
