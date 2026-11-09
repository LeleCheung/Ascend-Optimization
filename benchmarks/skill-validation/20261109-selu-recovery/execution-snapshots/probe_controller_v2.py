import fcntl,json,shlex,subprocess
from pathlib import Path
b=Path('/data/hanle/ascend-optimization');r=b/'runs/selu-skill-recovery-20261109-v2';stage=b/'runs/.queued/selu-skill-recovery-20261109-v2'
cmd=['/usr/local/python3.11.15/bin/python3.11',str(stage/'package/tools/skill_validation/probe_launch.py'),
 '--source',str(r/'workspace/starter.py'),'--leaf',str(r/'reference-catalog'),
 '--output',str(r/'baseline-probe-v2'),]
shell='source /usr/local/Ascend/cann-9.0.0/set_env.sh\n'
shell+='export ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\n'
shell+='export TRITON_CACHE_DIR='+shlex.quote(str(r/'probe-triton-cache'))+'\n'
shell+='cd '+shlex.quote(str(r))+'\nexec '+shlex.join(cmd)
with (b/'runs/physical-card-7-campaign.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 with (r/'baseline-probe-v2.log').open('xb') as out:
  p=subprocess.run(['docker','exec','tle_yy','bash','-c',shell],stdout=out,stderr=subprocess.STDOUT,timeout=1200)
(r/'state.json').write_text(json.dumps(dict(status='probe_completed' if p.returncode==0 else 'requires_attention',returncode=p.returncode))+'\n')
