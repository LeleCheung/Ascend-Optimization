from pathlib import Path
import subprocess,json,os,shlex,fcntl,hashlib
r=Path('/data/hanle/ascend-optimization');w=r/'runs/flaggems-upstream-log10-20261111';repo=w/'FlagGems-baseline'
upstream=w/'FlagGems-filtered';host=w/'host-venv/bin/python';out=w/'reference-native'
expected='d6a8eec473517a3d68157b208eb9c057eb1d4c50'
subprocess.run(['git','-C',str(upstream),'worktree','add','--detach',str(repo),expected],check=True,capture_output=True)
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain=v1'],text=True).strip()
out.mkdir()
python=r/'runtime/flaggems-validation-venv/bin/python'
c=dict(repository=str(repo),operator='log10_',accuracy_files=['tests/test_log10.py'],benchmark_files=['benchmark/test_log10.py'],
 accuracy_reference='cpu',timeout_seconds=3600)
(out/'config.json').write_text(json.dumps(c,indent=2))
runtime=Path(subprocess.check_output([str(host),'-c','import hgkda.runtime.upstream_driver as m;print(m.__file__)'],text=True).strip()).parent
env=dict(ASCEND_RT_VISIBLE_DEVICES='7',NPU_VISIBLE_DEVICES='7',PYTHONPATH=str(repo/'src'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',TRITON_CACHE_DIR=str(out/'triton-cache'))
def device(label,args):
 shell='source /usr/local/Ascend/cann-9.0.0/set_env.sh'+chr(10)+chr(10).join('export '+k+'='+shlex.quote(v) for k,v in env.items())+chr(10)+'exec '+shlex.join(list(map(str,args)))
 with (out/(label+'.log')).open('xb') as log:
  p=subprocess.run(['docker','exec','-w',str(repo),'tle_yy','bash','-c',shell],stdout=log,stderr=subprocess.STDOUT,timeout=3600)
 assert p.returncode==0,label
with (r/'runs/physical-card-7-campaign.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 device('import',[python,runtime/'upstream_probe.py','--repository',repo,'--operator','log10_','--output',out/'import.json'])
 identity=json.loads((out/'import.json').read_bytes());assert identity['source']=='src/flag_gems/ops/log10.py'
 device('tests',[python,runtime/'upstream_driver.py','--config',out/'config.json','--output',out/'upstream'])
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==expected
assert not subprocess.check_output(['git','-C',str(repo),'diff','HEAD'],text=True).strip()
print('original FlagGems reference completed',flush=True)
