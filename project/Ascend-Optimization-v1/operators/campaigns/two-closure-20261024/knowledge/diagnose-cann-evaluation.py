#!/usr/bin/env python3
"""在本批KG搜索空闲边界暂停己方任务，独立进程串行验证候选，始终恢复。"""
from pathlib import Path
import datetime,hashlib,json,os,signal,time,urllib.request

ROOT=Path('/data/hanle/ascend-optimization')
BATCH=ROOT/'Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024'
WORKSPACE=ROOT/'runs/two-closure-20261024/kg-runs/upsample_bilinear2d-two-closure-20261024-attempt-1'

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
 out=BATCH/'research/cann-eval-control';out.mkdir(parents=True,exist_ok=False)
 state=dict(status='waiting_for_search_boundary',jobs={});save(out/'control.json',state)
 end=time.monotonic()+1800
 while True:
  campaign=read(BATCH/'endurance-state.json')
  if (campaign.get('current_job')=='search:upsample_bilinear2d:1' and read(BATCH/'research/joined-probes/control.json').get('status')=='resumed' and idle()):break
  assert campaign['status'] not in ['completed','needs_attention'],campaign['status']
  if time.monotonic()>end:raise TimeoutError('未找到安全搜索边界；未暂停任何进程')
  time.sleep(2)
 controllers=[];groups=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:args,start,cwd=proc(int(p.name))
  except (OSError,UnicodeDecodeError):continue
  pid=int(p.name)
  if any(a.endswith('/bootstrap-two-closure.py') for a in args) and BATCH.name in args:controllers.append(dict(pid=pid,start=start))
  if args and 'claude' in Path(args[0]).name and cwd.startswith(str(WORKSPACE)+'/') and os.getsid(pid)==pid:
   assert os.getpgid(pid)==pid and pid!=os.getpgrp();groups.append(dict(pid=pid,start=start,cwd=cwd))
 assert len(controllers)==1 and len(groups)<=4,(controllers,groups)
 state.update(status='preparing',controllers=controllers,claude_groups=groups);save(out/'control.json',state)
 stopped=[]
 try:
  for p in controllers:
   assert proc(p['pid'])[1]==p['start'];os.kill(p['pid'],signal.SIGSTOP);stopped.append((p,False))
  assert read(BATCH/'endurance-state.json')['current_job']=='search:upsample_bilinear2d:1'
  for p in groups:
   assert proc(p['pid'])[1]==p['start'];os.killpg(p['pid'],signal.SIGSTOP);stopped.append((p,True))
  end=time.monotonic()+60
  while not idle():
   if time.monotonic()>end:raise RuntimeError('在途设备任务未退出，不提交探针')
   time.sleep(2)
  state['status']='own_tasks_paused';save(out/'control.json',state)
  for name in ['async_probe','deterministic_complete']:
   folder=out/name;folder.mkdir(exist_ok=False)
   source=(BATCH/'knowledge'/('probe-cann-hybrid-async.py' if name=='async_probe' else 'evaluate-cann-deterministic.py')).read_text();(folder/'probe.py').write_text(source)
   request=dict(command=['{python}','probe.py'],files=[dict(path='probe.py',content=source)],timeout_seconds=1200)
   save(folder/'request.json',request)
   state['current_job']=name;save(out/'control.json',state)
   assert idle()
   response=json.load(opener.open(urllib.request.Request('http://127.0.0.1:19659/debug/jobs',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'},method='POST'),timeout=request['timeout_seconds']+45))
   save(folder/'response.json',response)
   state['jobs'][name]=dict(status=response['status'],job_id=response['job_id']);save(out/'control.json',state)
   assert idle(),'KGS已知结果后健康状态未知，停止后续请求'
  text='# 连续输出拼接与CANN扫描独立探针\n\n'
  for name,job in state['jobs'].items():
   text+=name+'：'+job['status']+'，job '+job['job_id']+'。'
   timings=out/name/'timings.json'
   if timings.exists():text+=json.dumps(read(timings),ensure_ascii=False)
   else:
    response=read(out/name/'response.json');text+=response.get('stderr','')[-1700:]
   text+='\n\n'
  (BATCH/'knowledge/连续拼接与原生扫描实测.md').write_text(text)
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
 print(json.dumps(state,ensure_ascii=False))

if __name__=='__main__':execute()
