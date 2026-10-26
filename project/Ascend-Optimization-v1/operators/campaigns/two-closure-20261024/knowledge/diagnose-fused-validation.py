#!/usr/bin/env python3
"""在本批KG搜索空闲边界暂停己方任务，独立进程串行验证候选，始终恢复。"""
from pathlib import Path
import datetime,hashlib,json,os,signal,time,urllib.request,fcntl,subprocess,sys

ROOT=Path('/data/hanle/ascend-optimization')
BATCH=ROOT/'Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024'
WORKSPACE=ROOT/'runs/two-closure-20261024/kg-runs/upsample_bilinear2d-two-closure-20261024-'

def read(p):return json.loads(p.read_text())
def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def proc(pid):
 p=Path('/proc')/str(pid);stat=p.joinpath('stat').read_text().rpartition(')')[2].split()
 return p.joinpath('cmdline').read_bytes().decode().split('\0'),stat[19],str(p.joinpath('cwd').resolve())

def execute():
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 def idle():
  for port in [19655,19657,19659]:
   d=json.load(opener.open('http://127.0.0.1:%d/status'%port,timeout=15))['scheduler']
   if d['healthy']!=1 or any(d[k] for k in ['active','waiting','broken']):return False
  return True
 out=BATCH/'research/fused-validation-control';out.mkdir(parents=True,exist_ok=False)
 state=dict(status='waiting_for_search_boundary',jobs={});save(out/'control.json',state)
 end=time.monotonic()+3600
 standalone_lock=None
 while True:
  campaign=read(BATCH/'endurance-state.json')
  predecessor=BATCH/'research/scan-launch-validation-control-v2/control.json'
  ready=predecessor.exists() and read(predecessor).get('status')=='resumed'
  if ready and campaign.get('current_job') in ['search:upsample_bilinear2d:2']:
   heartbeat=read(BATCH/'heartbeat.json');metadata=Path(heartbeat['workspace'])/'.kernelgen/run-process.json'
   found_group=False
   for candidate in Path('/proc').iterdir():
    if not candidate.name.isdigit():continue
    try:
     args,start,cwd=proc(int(candidate.name))
     if args and 'claude' in Path(args[0]).name and cwd.startswith(str(WORKSPACE)) and os.getsid(int(candidate.name))==int(candidate.name):found_group=True
    except (OSError,UnicodeDecodeError):continue
   if metadata.exists() and not read(metadata).get('finished_at') and found_group and idle():break
  if ready and campaign.get('status')=='completed' and idle():
   standalone_lock=(ROOT/'runs/physical-card-7-campaign.lock').open('a')
   fcntl.flock(standalone_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  assert campaign['status']!='needs_attention',campaign['status']
  if time.monotonic()>end:raise TimeoutError('未找到安全搜索边界；未暂停任何进程')
  time.sleep(2)
 controllers=[];groups=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:args,start,cwd=proc(int(p.name))
  except (OSError,UnicodeDecodeError):continue
  pid=int(p.name)
  if any(a.endswith('/resume-two-closure.py') for a in args) and BATCH.name in args:controllers.append(dict(pid=pid,start=start))
  if args and 'claude' in Path(args[0]).name and cwd.startswith(str(WORKSPACE)) and os.getsid(pid)==pid:
   assert os.getpgid(pid)==pid and pid!=os.getpgrp();groups.append(dict(pid=pid,start=start,cwd=cwd))
 assert (standalone_lock is not None and not controllers and not groups) or (len(controllers)==1 and 1<=len(groups)<=4),(controllers,groups)
 state.update(status='preparing',controllers=controllers,claude_groups=groups);save(out/'control.json',state)
 stopped=[]
 try:
  for p in controllers:
   assert proc(p['pid'])[1]==p['start'];os.kill(p['pid'],signal.SIGSTOP);stopped.append((p,False))
  assert (standalone_lock is not None and read(BATCH/'endurance-state.json')['status']=='completed') or read(BATCH/'endurance-state.json')['current_job']==campaign['current_job']
  for p in groups:
   assert proc(p['pid'])[1]==p['start'];os.killpg(p['pid'],signal.SIGSTOP);stopped.append((p,True))
  end=time.monotonic()+60
  while not idle():
   if time.monotonic()>end:raise RuntimeError('在途设备任务未退出，不提交探针')
   time.sleep(2)
  state['status']='own_tasks_paused';save(out/'control.json',state)
  for name in ['corrected_fused']:
   folder=out/name;folder.mkdir(exist_ok=False)
   tool=BATCH.parents[2]/'tools/profiling/hgkda/validate-two-fused.py'
   save(folder/'request.json',dict(tool=str(tool),sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),scope='宿主控制器；内部串行提交KGS准入和评测，避免嵌套占用同一设备槽'))
   state['current_job']=name;save(out/'control.json',state)
   completed=subprocess.run([sys.executable,str(tool)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=1500)
   response=dict(status='SUCCEEDED' if completed.returncode==0 else 'FAILED',job_id='host-controller',exit_code=completed.returncode,stdout=completed.stdout.decode(errors='replace'),stderr=completed.stderr.decode(errors='replace'))
   save(folder/'response.json',response)
   state['jobs'][name]=dict(status=response['status'],job_id=response['job_id']);save(out/'control.json',state)
   assert idle(),'KGS已知结果后健康状态未知，停止后续请求'
  text='# 独立进程复核相邻行复用候选\n\n'
  for name,job in state['jobs'].items():
   text+=name+'：'+job['status']+'，job '+job['job_id']+'。'
   timings=out/name/'timings.json'
   if timings.exists():text+=json.dumps(read(timings),ensure_ascii=False)
   else:
    response=read(out/name/'response.json');text+=response.get('stderr','')[-1700:]
   text+='\n\n'
  (BATCH/'knowledge/融合双输出行独立验收.md').write_text(text)
 finally:
  resumed=[]
  for p,group in reversed(stopped):
   try:
    assert proc(p['pid'])[1]==p['start']
    if group:os.killpg(p['pid'],signal.SIGCONT)
    else:os.kill(p['pid'],signal.SIGCONT)
    resumed.append(p['pid'])
   except (OSError,AssertionError):resumed.append(dict(pid=p['pid'],status='identity_changed'))
  state.update(status='resumed',resumed=resumed,finished_at=datetime.datetime.utcnow().isoformat()+'Z');save(out/'control.json',state)
 if standalone_lock is not None:standalone_lock.close()
 print(json.dumps(state,ensure_ascii=False))

if __name__=='__main__':execute()
