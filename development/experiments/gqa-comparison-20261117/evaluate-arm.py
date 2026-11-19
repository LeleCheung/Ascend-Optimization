import argparse,json,subprocess,shlex,sys,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('arm');p.add_argument('round');a=p.parse_args()
r=Path('/data/hanle/ascend-optimization/runs/gqa-attribution-20261117');run=r/a.arm
assert a.arm in ['no-simulator-v2','with-simulator-v2']
source=run/'workspace/candidate.py';digest=hashlib.sha256(source.read_bytes()).hexdigest()
sys.path.insert(0,str(r/'method/hgkda/src'))
from hgkda.gate import admit
from hgkda.runtime.trace import action_trace
review=run/'source-reviews'/(digest+'.json');gate=admit(source.read_text(),json.loads(review.read_bytes()))
out=run/('evaluation-'+a.round);out.mkdir(exist_ok=False)
(out/'source.py').write_bytes(source.read_bytes());(out/'gate.json').write_text(json.dumps(gate,indent=2))
cmd=['/usr/local/python3.11.15/bin/python3.11',str(r/'attention_bench.py'),'--contract',str(r/'contract.json'),'--candidate',str(out/'source.py'),'--output',str(out/'results.json')]
shell='source /usr/local/Ascend/cann-9.0.0/set_env.sh; export ASCEND_RT_VISIBLE_DEVICES=7; export TRITON_CACHE_DIR='+str(out/'cache')+'; exec '+shlex.join(cmd)
config=json.loads((run/'config.json').read_bytes());config['trace_round_id']=a.round
with action_trace(config,'attention_subset_benchmark',a.round,[out/'results.json',out/'gate.json',r/'contract.json']):
    with (out/'run.log').open('w') as log:
        result=subprocess.run(['flock','/data/hanle/ascend-optimization/runs/physical-card-7-campaign.lock','docker','exec','tle_yy','bash','-c',shell],stdout=log,stderr=subprocess.STDOUT,timeout=900)
    (out/'execution.json').write_text(json.dumps({'returncode':result.returncode,'source_sha256':digest}))
print(a.arm,result.returncode)
