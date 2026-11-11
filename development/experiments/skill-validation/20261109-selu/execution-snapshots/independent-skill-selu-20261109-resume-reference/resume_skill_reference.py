"""保留成功采集，仅按真实kernel名称补采参考侧，随后启动受控模型阶段。"""
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

base=Path('/data/hanle/ascend-optimization')
repo=base/'Ascend-Optimization'
run=base/'runs/independent-skill-selu-20261109-v1'
config=json.loads((run/'config.json').read_bytes())
side=run/'baseline-paired/reference'
kernel='Selu_d2db1a80c523e7e59a032c95969880af_high_performance_210010000'
assert not (side/'collection.json').exists()
application=shlex.join([config['device_python'],str(repo/'tools/skill_validation/paired_driver.py'),
                      '--spec',str(run/'baseline-paired/spec.json'),'--side','reference'])
command=['msprof','op','--application='+application,'--output='+str(side/'profile-corrected'),
         '--kernel-name='+kernel,'--launch-count=1','--warm-up=5','--aic-metrics=Roofline']
shell='source '+shlex.quote(str(Path(config['sdk'])/'set_env.sh'))+'\n'
environment=dict(ASCEND_RT_VISIBLE_DEVICES=str(config['device']),OMP_NUM_THREADS='1',
                 PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(repo/'third_party/FlagGems/src'),
                 TRITON_CACHE_DIR=str(run/'profiling-triton-cache'))
shell+='\n'.join('export '+k+'='+shlex.quote(v) for k,v in environment.items())+'\nexec '+shlex.join(command)
with Path(config['lock']).open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    with (run/'collect-reference-corrected.log').open('xb') as output:
        subprocess.run(['docker','exec',config['container'],'bash','-c',shell],stdout=output,
                       stderr=subprocess.STDOUT,timeout=900,check=True)
    basic=list((side/'profile-corrected').rglob('OpBasicInfo.csv'))
    assert len(basic)==1,basic
    with basic[0].open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==1 and rows[0]['Op Name']==kernel,rows
    record=dict(status='completed',side='reference',expected_kernel=kernel,groups='Roofline',
         command=command,spec_sha256=hashlib.sha256((run/'baseline-paired/spec.json').read_bytes()).hexdigest(),
         driver_sha256=hashlib.sha256((repo/'tools/skill_validation/paired_driver.py').read_bytes()).hexdigest(),
         correction='从原始采集日志确认本机CANN Selu名称；先前错误Elu筛选日志保留')
    (side/'collection.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
subprocess.run([sys.executable,str(repo/'tools/skill_validation/agent_runner.py'),
               '--config',str(run/'config.json'),'--model-settings','/root/.claude/settings.json'],check=True)
