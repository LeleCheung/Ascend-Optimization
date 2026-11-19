from pathlib import Path
import subprocess,shlex,json,hashlib
r=Path('/data/hanle/ascend-optimization/runs/gqa-attribution-20261117')
source=Path('/data/hanle/ascend-optimization/runs/night-research-20261116/gqa/workspace/candidate.py')
assert hashlib.sha256(source.read_bytes()).hexdigest()=='770910b7b720dd72ebf498d1baec91d0e254f032d1201da6316747bf290ca585'
cmd=['/usr/local/python3.11.15/bin/python3.11',str(r/'attention_bench.py'),'--contract',str(r/'contract.json'),'--candidate',str(source),'--output',str(r/'official-check-01.json')]
inside='source /usr/local/Ascend/cann-9.0.0/set_env.sh; export ASCEND_RT_VISIBLE_DEVICES=7; export TRITON_CACHE_DIR='+str(r/'cache')+'; exec '+shlex.join(cmd)
with (r/'official-check-01.log').open('w') as log:
    p=subprocess.Popen(['flock','/data/hanle/ascend-optimization/runs/physical-card-7-campaign.lock','docker','exec','tle_yy','bash','-c',inside],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(r/'official-check-process.json').write_text(json.dumps({'pid':p.pid,'command':cmd}))
print(p.pid)
