"""Freeze real-device autotuning, then replay the selected configuration in simulation."""
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

r=Path('/data/hanle/ascend-optimization/runs/native-upsample-contrast-20261118');run=r/'native'
out=run/(sys.argv[1] if len(sys.argv)>1 else 'selected-replay');out.mkdir(exist_ok=False)
spec=json.loads((run/'paired/spec.json').read_bytes());spec['ir_directory']=str(out/'actual-ir')
(out/'actual-ir').mkdir()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
save(out/'real-spec.json',spec)
driver=r/'paired_driver_replay.py'
base='source /usr/local/Ascend/cann-9.0.0/set_env.sh\nexport ASCEND_RT_VISIBLE_DEVICES=7\nexport TRITON_DISABLE_LINE_INFO=false\nexport PYTHONPATH=/data/hanle/ascend-optimization/Ascend-Optimization/development/dependencies/third_party/FlagGems/src\nexport TRITON_CACHE_DIR='+str(out/'cache')+'\n'
def execute(label,cmd,sim=False):
    shell=base
    if sim:shell+='export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/aarch64-linux/simulator/Ascend910B4/lib:"${LD_LIBRARY_PATH:-}"\n'
    with (r.parent/'physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with (out/(label+'.log')).open('x') as log:
            proc=subprocess.run(['docker','exec','-w',str(out),'tle_yy','bash','-c',shell+'exec '+shlex.join(cmd)],stdout=log,stderr=subprocess.STDOUT,timeout=450)
    save(out/(label+'-execution.json'),{'returncode':proc.returncode,'command':cmd})
    return proc.returncode
assert execute('real',['/usr/local/python3.11.15/bin/python3.11',str(driver),'--spec',str(out/'real-spec.json'),'--side','candidate'])==0
cache=out/'actual-ir/autotune-cache.json';record=json.loads(cache.read_bytes());assert record['entries'] and all(e['cache'] for e in record['entries'])
spec.update(autotune_replay=str(cache),warmup_before_capture=0,ir_directory=str(out/'sim-ir'))
save(out/'sim-spec.json',spec)
app=shlex.join(['/usr/local/python3.11.15/bin/python3.11',str(driver),'--spec',str(out/'sim-spec.json'),'--side','candidate'])
cmd=['msprof','op','simulator','--application='+app,'--kernel-name=upsample_bilinear2d_kernel','--soc-version=Ascend910B4','--core-id=0','--launch-count=1','--dump=on','--timeout=1','--output='+str(out/'raw')]
save(out/'request.json',{'command':cmd,'source_sha256':spec['source_sha256'],'case':spec['case'],'autotune_cache_sha256':hashlib.sha256(cache.read_bytes()).hexdigest(),'scope':'selected real-device autotune configuration replay; bounded observed window'})
execute('simulator',cmd,True)
print(out)
