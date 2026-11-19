from pathlib import Path
import subprocess,shlex,json,sys,hashlib
r=Path('/data/hanle/ascend-optimization/runs/gqa-attribution-20261117')
run=r/'with-simulator-v2';source=run/'evaluation-round-02/source.py'
result=json.loads((run/'evaluation-round-02/results.json').read_bytes())
assert all(x['candidate_correctness']['official_checker']['passed'] for x in result['cases'])
assert hashlib.sha256(source.read_bytes()).hexdigest()==result['source_sha256']
sys.path.insert(0,str(r/'method/hgkda/src'))
from hgkda.runtime.trace import action_trace
config=json.loads((run/'config.json').read_bytes());config['trace_round_id']='round-02'
for name,sim in [('candidate',False),('simulator',True)]:
    out=r/'capture-after'/name;out.mkdir(parents=True,exist_ok=False)
    app=shlex.join(['/usr/local/python3.11.15/bin/python3.11',str(r/'attention_profile_driver.py'),'--contract',str(r/'contract.json'),'--source',str(source),'--case','4','--side','candidate','--output',str(out/'actual-ir')])
    if sim:
        cmd=['msprof','op','simulator','--application='+app,'--kernel-name=_gqa_fwd_kernel','--soc-version=Ascend910B4','--core-id=0','--launch-count=1','--dump=on','--timeout=3','--output='+str(out/'raw')]
    else:cmd=['msprof','op','--application='+app,'--kernel-name=_gqa_fwd_kernel','--launch-count=1','--warm-up=2','--aic-metrics=Roofline','--output='+str(out/'raw')]
    shell='source /usr/local/Ascend/cann-9.0.0/set_env.sh\nexport ASCEND_RT_VISIBLE_DEVICES=7\nexport TRITON_DISABLE_LINE_INFO=false\nexport TRITON_CACHE_DIR='+str(out/'cache')+'\n'
    if sim:shell+='export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/aarch64-linux/simulator/Ascend910B4/lib:${LD_LIBRARY_PATH:-}\n'
    shell+='exec '+shlex.join(cmd)
    (out/'request.json').write_text(json.dumps({'command':cmd,'source_sha256':result['source_sha256'],'case_id':4},indent=2))
    with action_trace(config,'simulate' if sim else 'pair','after-round-02',[out/'request.json',out/'execution.json']):
        with (out/'collector.log').open('w') as log:
            done=subprocess.run(['flock','/data/hanle/ascend-optimization/runs/physical-card-7-campaign.lock','docker','exec','-w',str(out),'tle_yy','bash','-c',shell],stdout=log,stderr=subprocess.STDOUT,timeout=360)
        (out/'execution.json').write_text(json.dumps({'returncode':done.returncode}))
    print(name,done.returncode,flush=True)
