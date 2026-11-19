"""Verify a retained starting point, then collect paired and explicitly selected simulator evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import shlex

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--public-package',type=Path,required=True)
p.add_argument('--validated-start',action='store_true',help='Check the already reviewed and formally validated delivery without rewriting its dependencies')
p.add_argument('--resume-pair',action='store_true',help='Reuse validated starting evidence and resume incomplete profiling')
a=p.parse_args();r=a.run;read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
c=read(r/'config.json');digest=sha(r/'workspace/candidate.py');formal=r/('formal-'+digest[:12])
pid=read(r/'starting-validation-process.json')['pid']
while not (formal/'audit-final.json').exists():
    if not Path('/proc/'+str(pid)).exists():
        raise RuntimeError('Starting evaluation exited; inspect starting-validation.log')
    time.sleep(5)
audit=read(formal/'audit-final.json')['rows'][0]
assert audit['source_sha256']==digest and audit['full_passed'] and all(x['status']=='passed' for x in audit['checks'].values())
package=Path(c['package'])
assert not (r/'agent-calls').exists(), 'Freeze collection update before the first model invocation'
shutil.copyfile(r/'method-manifest.json',r/'method-manifest-before-multikernel.json')
for name in ['collect_pair.py','collect_simulator.py']:
    shutil.copyfile(a.public_package/'src/hgkda/runtime'/name,package/'src/hgkda/runtime'/name)
save(r/'method-manifest.json',dict(files=[dict(path=str(p.relative_to(package)),sha256=sha(p)) for p in package.rglob('*') if p.is_file() and '__pycache__' not in p.parts]))
env=dict(os.environ,PYTHONPATH=str(package/'src'))
state=dict(source_sha256=digest,status='collecting')
def execute(label,argv):
    log_name=label+('-resume' if a.resume_pair else '')
    with (r/(log_name+'.log')).open('x') as log:
        rc=subprocess.run([sys.executable,*argv],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    state[label]=rc;save(r/'starting-evidence-status.json',state);return rc
if a.resume_pair:
    verification=read(r/'attempts'/digest/'compiled/verification.json')
    assert verification['status']=='verified' and verification['source_sha256']==digest
elif a.validated_start:
    sys.path.insert(0,str(package/'src'))
    from hgkda.gate import admit
    import fcntl
    admit((formal/'candidate.py').read_text(),read(r/'source-reviews'/(digest+'.json')))
    attempt=r/'attempts'/digest;attempt.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(formal/'candidate.py',attempt/'candidate.py')
    environment=dict(ASCEND_RT_VISIBLE_DEVICES=str(c['device']),PYTHONPATH=c.get('device_pythonpath',''),
        OMP_NUM_THREADS='1',TRITON_CACHE_DIR=str(attempt/'triton-cache'),PYTHONDONTWRITEBYTECODE='1')
    argv=[c['device_python'],str(package/'src/hgkda/runtime/capture_compiled.py'),'--source',str(attempt/'candidate.py'),
        '--leaf',c['job']['checks']['supplementary']['leaf'],'--output',str(attempt/'compiled')]
    shell='source '+shlex.quote(c['sdk']+'/set_env.sh')+'\n'
    shell+='\n'.join('export '+k+'='+shlex.quote(v) for k,v in environment.items())+'\nexec '+shlex.join(argv)
    with Path(c['lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with (r/'starting-compiled-reviewed.log').open('x') as log:
            subprocess.run(['docker','exec',c['container'],'bash','-c',shell],stdout=log,stderr=subprocess.STDOUT,check=True)
    verification=read(attempt/'compiled/verification.json')
    assert verification['status']=='verified' and verification['source_sha256']==digest
    save(r/'starting-reviewed-qualification.json',dict(source_sha256=digest,formal=str(formal/'audit-final.json'),
        compiled=str(attempt/'compiled/verification.json'),scope='Fixed reviewed delivery baseline; new model candidates still use normal preflight imports and checks.'))
else:
    assert execute('starting-preflight',['-m','hgkda.runtime.agent_tools','--config',str(r/'config.json'),'preflight','--source',str(r/'workspace/candidate.py')])==0
    assert read(r/'workspace/latest-preflight.json')['status']=='passed'
result=read(formal/'evaluation/result.json');target=next(x for x in result['per_workload'] if x['uuid']==c['experiment_goal']['target_case'])
best=r/'best';best.mkdir(exist_ok=a.resume_pair);shutil.copyfile(formal/'candidate.py',best/'candidate.py')
save(best/'selection.json',dict(source_sha256=digest,target_latency_ms=target['latency_ms'],target_speedup=target['speedup'],geo_mean=result['geo_mean'],validation=str(formal),origin='retained delivery remeasured in current environment'))
pair=r/('paired-'+digest[:8]);sim=r/('simulator-'+digest[:8])
assert execute('starting-pair',['-m','hgkda.runtime.collect_pair','--config',str(r/'config.json'),'--source',str(formal/'candidate.py'),'--case',c['experiment_goal']['target_case'],'--output',str(pair),*(['--resume'] if a.resume_pair else [])])==0
discovery=read(pair/'candidate-discovery.json');durations={}
for task in discovery['tasks']:durations[task['name']]=durations.get(task['name'],0)+task['duration_us']
kernel=max(durations,key=durations.get)
save(r/'simulator-selection.json',dict(source_sha256=digest,kernel=kernel,observed_task_duration_us=durations,
    reason='Largest accumulated device-task duration in the same-case discovery trace; individual kernel capture, full operator measured separately.'))
execute('starting-simulator',['-m','hgkda.runtime.collect_simulator','--config',str(r/'config.json'),'--paired-directory',str(pair),'--output',str(sim),
    '--kernel-name',kernel,'--soc','Ascend910B4','--simulator-library',c['sdk']+'/aarch64-linux/simulator/Ascend910B4/lib',
    '--timeout-minutes','1','--trace-unit','us','--allow-postcapture-exit'])
summary_rc=execute('starting-summary',[str(r/'contrast_simulator_summary.py'),'--study',str(r),'--directory',str(sim)])
if summary_rc:
    assert execute('starting-instruction-summary',[str(r.parent/'summarize_instruction_observation.py'),'--run',str(r),'--directory',str(sim)])==0
assert execute('starting-timeline',[str(r.parent/'relative_timeline_report.py'),'--directory',str(sim),'--package',str(a.public_package)])==0
state['status']='evidence_ready';save(r/'starting-evidence-status.json',state)
print(json.dumps(state))
