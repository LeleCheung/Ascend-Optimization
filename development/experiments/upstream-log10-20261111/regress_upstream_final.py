from pathlib import Path
import json,subprocess,os,hashlib,shutil,shlex,socket,time,signal,urllib.request
r=Path('/data/hanle/ascend-optimization');w=r/'runs/flaggems-upstream-log10-20261111';project=r/'Ascend-Optimization'
host=w/'host-venv/bin/python';method=w/'method-v4/hgkda';work=w/'kgs-final'
def read(p):return json.loads(p.read_bytes())
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
def run(label,args):
 with (w/(label+'.log')).open('wb') as log:p=subprocess.run(list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,timeout=2400)
 print(label,p.returncode,flush=True)
 if p.returncode:raise RuntimeError(label+' failed; inspect log')
assert read(w/'attempt-3/result.json')['result']['status']=='passed'
port=19715
with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0
backend=r/'runtime/repository-migration-20261111/external-backends'
settings=dict(container='tle_yy',device=7,device_python='/usr/local/python3.11.15/bin/python3.11',
 sdk='/usr/local/Ascend/cann-9.0.0',lock=str(r/'runs/physical-card-7-campaign.lock'),kgs_server='http://127.0.0.1:'+str(port))
save(w/'kgs-environment-final.json',settings)
env=dict(ASCEND_RT_VISIBLE_DEVICES='7',KGS_FLAGGEMS_ROOT=str(backend/'FlagGems'),
 PYTHONPATH=':'.join(map(str,[backend/'kernelgen_server/client',backend/'kernelgen_server',backend])),
 NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost',PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=str(w/'kgs-cache'))
command=[settings['device_python'],'-u','-m','kernelgen_server.server','--host','127.0.0.1','--backend','npu','--timing','walltime',
 '--max-workers','1','--port',str(port),'--profile-artifact-root',str(w/'kgs-profiles')]
shell='source '+settings['sdk']+'/set_env.sh'+chr(10)+chr(10).join('export '+k+'='+shlex.quote(v) for k,v in env.items())+chr(10)+'exec '+shlex.join(command)
(w/'kgs-service-final.sh').write_text(shell)
log=(w/'kgs-service-final.log').open('xb');proc=subprocess.Popen(['docker','exec','-w',str(w),'tle_yy','bash',str(w/'kgs-service-final.sh')],stdout=log,stderr=subprocess.STDOUT)
owned=None
try:
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 for _ in range(90):
  if proc.poll() is not None:raise RuntimeError('KGS startup failed')
  try:
   with opener.open(settings['kgs_server']+'/status',timeout=2) as response:status=json.load(response)
   break
  except Exception:time.sleep(1)
 else:raise RuntimeError('KGS startup timeout')
 owners=[]
 for p in Path('/proc').glob('[0-9]*'):
  try:
   cmd=(p/'cmdline').read_bytes().split(b'\0')
   if b'kernelgen_server.server' in cmd and b'--port' in cmd and cmd[cmd.index(b'--port')+1]==str(port).encode():owners.append(p)
  except (OSError,ValueError,IndexError):pass
 assert len(owners)==1
 owned=owners[0];start=(owned/'stat').read_text().split()[21];save(w/'kgs-service-final-identity.json',dict(pid=int(owned.name),starttime=start))
 run('kgs-prepare-final',[host,'-m','hgkda.runtime','prepare','--operator-directory',project/'operators/log10_',
    '--package',method,'--environment',w/'kgs-environment-final.json','--run-directory',work])
 source=w/'agent-abi-fix/log10.py';digest=hashlib.sha256(source.read_bytes()).hexdigest()
 shutil.copyfile(source,work/'workspace/candidate.py')
 shutil.copyfile(w/'review-final.json',work/'source-reviews'/(digest+'.json'))
 save(work/'workspace/starter-provenance.json',dict(kind='retained_candidate_with_model_profiler_compatibility_fix',
     source_sha256=digest,prior_best_source_provided=True,origin_sha256=read(w/'config.json')['delivery_sha256']))
 run('kgs-evaluate-final',[host,'-m','hgkda.runtime','evaluate','--config',work/'config.json'])
 audit=read(work/('formal-'+digest[:12])/'audit-final.json')
 save(w/'closure-final.json',dict(upstream=read(w/'attempt-3/result.json')['result'],kgs=audit,
     source_sha256=digest,scope='upstream integration and profiler compatibility; existing selected delivery unchanged'))
 print(json.dumps(audit),flush=True)
finally:
 if owned and owned.exists() and (owned/'stat').read_text().split()[21]==start:os.kill(int(owned.name),signal.SIGTERM)
 try:proc.wait(timeout=20)
 except subprocess.TimeoutExpired:proc.terminate()
 save(w/'kgs-service-final-stop.json',dict(project_service_only=True,stopped=True))
