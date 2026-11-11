from pathlib import Path
import subprocess,sys,os,json,hashlib,tarfile,shutil,socket,time,signal

root=Path('/data/hanle/ascend-optimization/runtime/repository-migration-20261111')
public=root/'public-final';repo=Path('/data/hanle/ascend-optimization/Ascend-Optimization')
read=lambda p:json.loads(p.read_bytes())
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def run(label,args,env=None,cwd=None,timeout=1800):
 log=root/(label+'.log')
 with log.open('wb') as f:p=subprocess.run(list(map(str,args)),cwd=cwd or public,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
 print(label,p.returncode,flush=True)
 if p.returncode:raise RuntimeError(label+': '+log.read_text(errors='replace')[-2300:])
if not public.exists():
 public.mkdir()
 assert hashlib.sha256((root/'public-final.tar.gz').read_bytes()).hexdigest()=='cd8e8ceb74162302b4a8e81676f55256fd1d986c11f2d4581244e37635aceba2'
 with tarfile.open(root/'public-final.tar.gz') as tar:tar.extractall(public,filter='data')
assert not (public/'development').exists()
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None)
if not (root/'wheels-final').exists():
 run('build-wheel',[sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--no-cache-dir','--wheel-dir',root/'wheels-final',public/'hgkda'],env)
python=root/'venv/bin/python'
if not python.exists():run('create-venv',[sys.executable,'-m','venv',root/'venv'],env)
wheel=next((root/'wheels-final').glob('*.whl'))
run('install',[python,'-m','pip','install','--no-deps','--no-cache-dir','--force-reinstall',wheel],env)
run('doctor',[python,'-m','hgkda','doctor'],env)
run('skill-doctor',[python,public/'hgkda/skills/hgkda/scripts/run_hgkda.py','doctor'],env)
save(root/'wheel.json',dict(name=wheel.name,sha256=hashlib.sha256(wheel.read_bytes()).hexdigest()))
service=None
try:
 # KGS remains an explicitly deployed external service, independent of development/.
 backend=root/'external-backends'
 if not backend.exists():
  backend.mkdir();shutil.copytree(repo/'sources/kernelgen_server',backend/'kernelgen_server',ignore=shutil.ignore_patterns('__pycache__'))
  shutil.copytree(repo/'third_party/FlagGems',backend/'FlagGems',ignore=shutil.ignore_patterns('__pycache__'))
 port=19713
 with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0,'port already owned'
 device_python='/usr/local/python3.11.15/bin/python3.11';sdk='/usr/local/Ascend/cann-9.0.0'
 settings=dict(container='tle_yy',device=7,device_python=device_python,sdk=sdk,
  lock='/data/hanle/ascend-optimization/runs/physical-card-7-campaign.lock',kgs_server='http://127.0.0.1:'+str(port))
 save(root/'environment.json',settings)
 import shlex
 service_env=dict(ASCEND_RT_VISIBLE_DEVICES='7',KGS_FLAGGEMS_ROOT=str(backend/'FlagGems'),
  PYTHONPATH=':'.join(str(x) for x in [backend/'kernelgen_server/client',backend/'kernelgen_server',backend]),
  NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost',TRITON_CACHE_DIR=str(root/'service-cache'),PYTHONDONTWRITEBYTECODE='1')
 command=[device_python,'-u','-m','kernelgen_server.server','--host','127.0.0.1','--backend','npu','--timing','walltime','--max-workers','1','--port',str(port),'--profile-artifact-root',str(root/'profiles')]
 shell='source '+sdk+'/set_env.sh\n'+'\n'.join('export '+k+'='+shlex.quote(v) for k,v in service_env.items())+'\nexec '+shlex.join(command)
 (root/'service.sh').write_text(shell)
 log=(root/'service.log').open('ab');service=subprocess.Popen(['docker','exec','-w',str(root),'tle_yy','bash',str(root/'service.sh')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 import urllib.request
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 for _ in range(60):
  if service.poll() is not None:raise RuntimeError('external KGS failed')
  try:
   with opener.open(settings['kgs_server']+'/status',timeout=2) as response:status=json.load(response)
   break
  except Exception:time.sleep(1)
 else:raise RuntimeError('KGS start timeout')
 save(root/'kgs-status.json',status)
 owned=[]
 for proc in Path('/proc').glob('[0-9]*'):
  try:
   cmd=(proc/'cmdline').read_bytes().split(b'\0')
   if b'kernelgen_server.server' in cmd and b'--port' in cmd and cmd[cmd.index(b'--port')+1]==str(port).encode():owned.append(int(proc.name))
  except (OSError,ValueError,IndexError):pass
 assert len(owned)==1,owned;save(root/'service-identity.json',dict(pid=owned[0],starttime=(Path('/proc')/str(owned[0])/'stat').read_text().split()[21],command=command))
 work=root/'validation-log10-final'
 if not work.exists():run('prepare',[python,'-m','hgkda.runtime','prepare','--operator-directory',public/'operators/log10_','--package',public/'hgkda','--environment',root/'environment.json','--run-directory',work],env)
 run('context',[python,'-m','hgkda.runtime','context','--config',work/'config.json'],env)
 run('preflight',[python,'-m','hgkda.runtime','preflight','--config',work/'config.json'],env)
 digest=hashlib.sha256((work/'workspace/candidate.py').read_bytes()).hexdigest();formal=work/('formal-'+digest[:12])
 if not formal.exists():run('evaluate',[python,'-m','hgkda.runtime','evaluate','--config',work/'config.json'],env)
 assert (formal/'audit-final.json').exists(),'evaluate incomplete; do not resubmit automatically'
 if not (work/'paired-migration').exists():run('pair',[python,'-m','hgkda.runtime.collect_pair','--config',work/'config.json','--source',work/'workspace/candidate.py','--case','log10_-timing-3','--output',work/'paired-migration'],env)
 save(root/'result.json',dict(status='passed',distribution_has_development=False,source_sha256=digest,wheel=read(root/'wheel.json'),
  formal=read(formal/'audit-final.json'),paired_comparison=str(work/'paired-migration/comparison.json'),
  scope='directory/runtime migration verification; fixed source; one complete KGS request; no replacement of delivery scores'))
 print('verification completed',flush=True)
finally:
 identity=root/'service-identity.json'
 if identity.exists():
  owner=read(identity);proc=Path('/proc')/str(owner['pid'])
  if proc.exists() and (proc/'stat').read_text().split()[21]==owner['starttime']:os.kill(owner['pid'],signal.SIGTERM)
 if service:
  try:service.wait(timeout=20)
  except subprocess.TimeoutExpired:service.terminate()
 save(root/'service-stop.json',dict(stopped=True,project_service_only=True))
