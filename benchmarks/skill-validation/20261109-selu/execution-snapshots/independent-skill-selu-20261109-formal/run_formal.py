import fcntl,json,shlex,subprocess,sys
from pathlib import Path
b=Path('/data/hanle/ascend-optimization');repo=b/'Ascend-Optimization';r=b/'runs/independent-skill-selu-20261109-v1'
p=json.loads((r/'workspace/latest-preflight.json').read_bytes())
command=['/usr/local/python3.11.15/bin/python3.11',str(repo/'tools/skill_validation/check_offset.py'),
 '--source',p['frozen_source'],'--output',str(r/'offset-check-final.json')]
shell='source /usr/local/Ascend/cann-9.0.0/set_env.sh\nexport ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\n'
shell+='export TRITON_CACHE_DIR='+shlex.quote(str(r/'offset-triton-cache'))+'\nexec '+shlex.join(command)
with (b/'runs/physical-card-7-campaign.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 with (r/'offset-check-final.log').open('xb') as out:
  subprocess.run(['docker','exec','tle_yy','bash','-c',shell],stdout=out,stderr=subprocess.STDOUT,timeout=300,check=True)
d=json.loads((r/'offset-check-final.json').read_bytes());assert d['passed']==d['total']==9,d
subprocess.run([sys.executable,str(repo/'tools/skill_validation/evaluate.py'),'--config',str(r/'config.json'),
 '--source',p['frozen_source'],'--review',str(r/'source-review.json'),'--port','19693'],check=True)
