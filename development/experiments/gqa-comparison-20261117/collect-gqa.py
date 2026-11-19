from pathlib import Path
import subprocess,shlex,json,os
r=Path('/data/hanle/ascend-optimization/runs/gqa-attribution-20261117')
source=Path('/data/hanle/ascend-optimization/runs/night-research-20261116/gqa/workspace/candidate.py')
python='/usr/local/python3.11.15/bin/python3.11'
for name,side,sim in [('candidate','candidate',False),('reference','reference',False),('simulator','candidate',True)]:
    out=r/'capture'/name;out.mkdir(parents=True,exist_ok=False)
    app=shlex.join([python,str(r/'attention_profile_driver.py'),'--contract',str(r/'contract.json'),'--source',str(source),'--case','4','--side',side,'--output',str(out/'actual-ir')])
    if sim:
        cmd=['msprof','op','simulator','--application='+app,'--kernel-name=_gqa_fwd_kernel','--soc-version=Ascend910B4','--core-id=0','--launch-count=1','--dump=on','--timeout=15','--output='+str(out/'raw')]
    else:
        cmd=['msprof','op','--application='+app,'--launch-count=1','--warm-up=2','--aic-metrics=Roofline','--output='+str(out/'raw')]
        if side=='candidate':cmd+=['--kernel-name=_gqa_fwd_kernel']
    shell='source /usr/local/Ascend/cann-9.0.0/set_env.sh\nexport ASCEND_RT_VISIBLE_DEVICES=7\nexport TRITON_DISABLE_LINE_INFO=false\nexport TRITON_CACHE_DIR='+str(out/'cache')+'\n'
    if sim:shell+='export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/aarch64-linux/simulator/Ascend910B4/lib:${LD_LIBRARY_PATH:-}\n'
    shell+='exec '+shlex.join(cmd)
    (out/'request.json').write_text(json.dumps({'command':cmd,'side':side,'case_id':4},indent=2))
    with (out/'collector.log').open('w') as log:
        try:
            result=subprocess.run(['flock','/data/hanle/ascend-optimization/runs/physical-card-7-campaign.lock','docker','exec','-w',str(out),'tle_yy','bash','-c',shell],stdout=log,stderr=subprocess.STDOUT,timeout=1050)
            status={'returncode':result.returncode}
        except subprocess.TimeoutExpired:status={'timeout':True}
    (out/'execution.json').write_text(json.dumps(status))
    print(name,status,flush=True)
