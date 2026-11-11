import fcntl,shlex,subprocess
from pathlib import Path
b=Path('/data/hanle/ascend-optimization');r=b/'runs/selu-skill-recovery-20261109-v2';stage=b/'runs/.queued/selu-skill-recovery-20261109-v2'
source=r/'attempts/0fed8d43398d2b3f1ceb50e55adea1bfc9fe906028f85145aa7cc9e0dce82626/candidate.py'
shell='source /usr/local/Ascend/cann-9.0.0/set_env.sh\nexport ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\n'
shell+='export TRITON_CACHE_DIR='+shlex.quote(str(source.parent/'triton-cache'))+'\nexec '+shlex.join(['/usr/local/python3.11.15/bin/python3.11',str(stage/'package/tools/skill_validation/probe_host.py'),'--source',str(source),'--output',str(r/'host-probe-0fed8d43398d.json')])
with (b/'runs/physical-card-7-campaign.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 with (r/'host-probe-0fed8d43398d.log').open('xb') as log:subprocess.run(['docker','exec','tle_yy','bash','-c',shell],stdout=log,stderr=subprocess.STDOUT,timeout=400,check=True)
