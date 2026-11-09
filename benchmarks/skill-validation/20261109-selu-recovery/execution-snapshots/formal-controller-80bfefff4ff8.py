import fcntl,json,shlex,subprocess,sys
from pathlib import Path
b=Path('/data/hanle/ascend-optimization');r=b/'runs/selu-skill-recovery-20261109-v2';stage=b/'runs/.queued/selu-skill-recovery-20261109-v2'
digest='80bfefff4ff837b1771eef1618199d56885ff3e950f02a6fc14dbe09ec3d3aca';source=r/'attempts'/digest/'candidate.py';checkroot=source.parent
with (b/'runs/physical-card-7-campaign.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for name in ['offset','runtime']:
  out=checkroot/(name+'-check.json')
  shell='source /usr/local/Ascend/cann-9.0.0/set_env.sh\nexport ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\n'
  shell+='export TRITON_CACHE_DIR='+shlex.quote(str(checkroot/'extra-triton-cache'))+'\nexec '+shlex.join(['/usr/local/python3.11.15/bin/python3.11',str(stage/'package/tools/skill_validation'/('check_'+name+'.py')),'--source',str(source),'--output',str(out)])
  with (checkroot/(name+'-check.log')).open('xb') as log:subprocess.run(['docker','exec','tle_yy','bash','-c',shell],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
  result=json.loads(out.read_bytes());assert result['passed']==result['total']
subprocess.run([sys.executable,str(stage/'package/tools/skill_validation/evaluate.py'),'--config',str(r/'config.json'),'--source',str(source),'--review',str(r/('source-review-'+digest[:12]+'.json')),'--port','19694'],check=True)
