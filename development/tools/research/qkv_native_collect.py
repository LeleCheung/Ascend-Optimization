"""Measure native QKV, then capture profiling and bounded simulator observations."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request

r=Path('/data/hanle/ascend-optimization/runs/qkv-simulator-20261118')
c=json.loads((r/'config.json').read_bytes());package=Path(c['package'])
sys.path.insert(0,str(package/'src'))
from hgkda.adapters.kgs import upload_bundle, call
save=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
for _ in range(90):
    try:
        status=call(c['kgs_server'],'/status');break
    except Exception:time.sleep(2)
else:raise RuntimeError('Dedicated KGS did not become ready')
save(r/'service-status.json',status)
with Path(c['lock']).open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    upload_bundle(c['kgs_server'],r/'operator.tar')
    request=json.loads((r/'native/request.json').read_bytes())
    request['implementation']['entrypoint']='main.py::run'
    request['implementation']['sources']=[dict(path='main.py',content='from upstream_native import _transform_bias_rescale_qkv as run\n'),
        dict(path='upstream_native.py',content=(r/'native/original.py').read_text())]
    save(r/'native/request-v2.json',request)
    result_path=r/'native/result.json'
    result=json.loads(result_path.read_bytes()) if result_path.exists() else call(c['kgs_server'],'/evaluate',request,timeout=2400,operation_id='qkv-native-20261118-v2')
save(r/'native/result.json',result)
env=dict(os.environ,PYTHONPATH=str(package/'src'))
pair=r/'native/paired'
resume=['--resume'] if pair.exists() else []
with (r/'native'/('pair-resume.log' if resume else 'pair.log')).open('x') as log:
    code=subprocess.run([sys.executable,'-m','hgkda.runtime.collect_pair','--config',str(r/'config.json'),
        '--source',str(r/'native/original.py'),'--case','transform_bias_rescale_qkv-timing-2','--output',str(pair),*resume],
        env=env,stdout=log,stderr=subprocess.STDOUT).returncode
assert code==0
out=r/'native/simulator';out.mkdir()
spec=json.loads((pair/'spec.json').read_bytes());spec['ir_directory']=str(out/'actual-ir');save(out/'spec.json',spec)
names=json.loads((pair/'candidate-discovery.json').read_bytes())['kernels'];assert len(names)==1
kernel=names[0]
app=shlex.join([c['device_python'],str(package/'src/hgkda/runtime/paired_driver.py'),'--spec',str(out/'spec.json'),'--side','candidate'])
cmd=['msprof','op','simulator','--application='+app,'--kernel-name='+kernel,'--soc-version=Ascend910B4',
    '--core-id=0','--launch-count=1','--dump=on','--timeout=1','--output='+str(out/'raw')]
save(out/'request.json',dict(command=cmd,source_sha256=spec['source_sha256'],case=spec['case'],kernel=kernel,
    scope='Raw fixed upstream reference; no generated candidate admission applied.'))
shell='source '+shlex.quote(c['sdk']+'/set_env.sh')+'\n'
vars=dict(ASCEND_RT_VISIBLE_DEVICES=str(c['device']),OMP_NUM_THREADS='1',TRITON_DISABLE_LINE_INFO='false',
    TRITON_CACHE_DIR=str(out/'compiler-cache'),PYTHONPATH=c['device_pythonpath'])
shell+='\n'.join('export '+k+'='+shlex.quote(v) for k,v in vars.items())+'\n'
shell+='export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/aarch64-linux/simulator/Ascend910B4/lib:"${LD_LIBRARY_PATH:-}"\nexec '+shlex.join(cmd)
with Path(c['lock']).open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    with (out/'collector.log').open('x') as log:
        code=subprocess.run(['docker','exec','-w',str(out),c['container'],'bash','-c',shell],stdout=log,stderr=subprocess.STDOUT,timeout=240).returncode
save(out/'execution.json',dict(returncode=code))
# The summarizer expects a method/hgkda source path, supplied as a local package link.
(r/'method').mkdir(exist_ok=True)
if not (r/'method/hgkda').exists():(r/'method/hgkda').symlink_to(package,target_is_directory=True)
with (out/'summary.log').open('x') as log:
    code=subprocess.run([sys.executable,str(r/'contrast_simulator_summary.py'),'--study',str(r),'--directory',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
save(r/'native/collection-status.json',dict(formal_status=result.get('status'),pair='completed',simulator_summary_returncode=code))
print(json.dumps(dict(formal_status=result.get('status'),summary_returncode=code)))
