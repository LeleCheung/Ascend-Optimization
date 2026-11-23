"""Collect source-bound profiling and a qualified simulator observation after validation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

r=Path(sys.argv[2]) if len(sys.argv)>2 else Path('/data/hanle/ascend-optimization/runs/upsample-performance-20261118')
old=r.parent/'native-upsample-contrast-20261118'
digest=sys.argv[1]
c=json.loads((r/'config.json').read_bytes())
source=r/('formal-'+digest[:12])/'candidate.py'
audit=json.loads((source.parent/'audit-final.json').read_bytes())
assert audit['rows'][0]['full_passed']
if not (r/'reference-catalog').exists():shutil.copytree(old/'with-simulator/reference-catalog',r/'reference-catalog')
for name in ['regions_optimized.py','contrast_simulator_summary.py']:
    if not (r/name).exists():shutil.copyfile(old/name,r/name)
env=dict(os.environ,PYTHONPATH=str(Path(c['package'])/'src'))
pair=r/('paired-'+digest[:8]);sim=r/('simulator-'+digest[:8])
record={}
def execute(name,argv):
    with (r/(name+'-'+digest[:8]+'.log')).open('x') as log:
        rc=subprocess.run([sys.executable,*argv],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    record[name]=rc
    (r/('capture-'+digest[:8]+'.json')).write_text(json.dumps(record,indent=2))
    return rc
assert execute('pair',['-m','hgkda.runtime.collect_pair','--config',str(r/'config.json'),
    '--source',str(source),'--case',c['experiment_goal']['target_case'],'--output',str(pair)])==0
discovery=json.loads((pair/'candidate-discovery.json').read_bytes())
selection=[]
if len(discovery['kernels'])>1:
    durations={}
    for task in discovery['tasks']:
        if task['task_type'].startswith('KERNEL_AI'):
            durations[task['name']]=durations.get(task['name'],0)+task['duration_us']
    kernel=max(durations,key=durations.get)
    selection=['--kernel-name',kernel]
    (r/('simulator-selection-'+digest[:8]+'.json')).write_text(json.dumps(dict(
        source_sha256=digest,kernel=kernel,observed_task_duration_us=durations,
        reason='Largest accumulated same-case AI-core task duration; other tasks remain in paired profiling.'),indent=2)+'\n')
execute('simulator',['-m','hgkda.runtime.collect_simulator','--config',str(r/'config.json'),
    '--paired-directory',str(pair),'--output',str(sim),'--soc','Ascend910B4',
    '--simulator-library','/usr/local/Ascend/cann-9.0.0/aarch64-linux/simulator/Ascend910B4/lib',
    '--timeout-minutes','1','--trace-unit','us','--allow-postcapture-exit',*selection])
# A bounded capture may leave partial raw evidence; the summarizer separately validates
# source, core and time units and explicitly labels its observed scope.
execute('summary',[str(r/'contrast_simulator_summary.py'),'--study',str(r),'--directory',str(sim)])
print(json.dumps(record))
