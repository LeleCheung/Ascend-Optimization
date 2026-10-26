#!/usr/bin/env python3
"""持卡锁先做新机制探针和独立验收，再连续执行KG/CC搜索。"""
import argparse,datetime,fcntl,hashlib,importlib.util,json,math,shlex,time,urllib.request
from pathlib import Path

TOOLS=Path(__file__).resolve().parent
ROOT=Path('/data/hanle/ascend-optimization')
PY='/usr/local/python3.11.15/bin/python3.11'

def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def read(p):return json.loads(p.read_text())
def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(p)

def execute(name):
 base=load('two_bootstrap_base',TOOLS/'run-two-closure.py')
 batch=ROOT/'Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns'/name
 path=batch/'bootstrap-state.json';assert not path.exists()
 state=dict(status='created',jobs={},time=datetime.datetime.utcnow().isoformat()+'Z');save(path,state)
 with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  campaign=base.Campaign(name)
  q=campaign.q
  campaign.step('baseline-version-check',lambda:campaign.run('verify-baseline',[PY,TOOLS/'verify-baseline.py','--server',q.server,'--output',batch/'baseline-verified.json'],600))
  assert campaign.state['jobs']['baseline-version-check']['status']=='completed'
  for kind,folder in [('upsample','initial-upsample'),('cumulative','reference-stability')]:
   out=batch/'research'/folder;out.mkdir(parents=True,exist_ok=False)
   script=(TOOLS/'probe-two-closure.py').read_text()
   request=dict(command=['{python}','probe.py','--batch-dir',str(batch),'--kind',kind],files=[dict(path='probe.py',content=script)],timeout_seconds=1200 if kind=='upsample' else 600)
   save(out/'request.json',request);(out/'probe.py').write_text(script)
   state.update(status='running',current_job='probe-'+kind);save(path,state)
   q.idle()
   try:
    response=json.load(q.opener.open(urllib.request.Request(q.server+'/debug/jobs',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'},method='POST'),timeout=request['timeout_seconds']+45))
    save(out/'response.json',response)
    state['jobs'][kind]=dict(status=response['status'],job_id=response['job_id']);save(path,state)
    q.idle()
   except Exception as error:
    state.update(status='needs_attention',error=type(error).__name__+': '+str(error));save(path,state);raise
  # 依据诊断延迟筛选独立候选，正式完整结果决定是否采用。
  data=read(batch/'research/initial-upsample/results.json')
  parent=next(r for r in data['rows'] if r['name']=='parent');assert parent['status']=='passed'
  times={r['case']:r['median_ms'] for r in parent['cases']}
  candidates=[]
  for row in data['rows']:
   if row['name']=='parent' or row['status']!='passed':continue
   assert {r['case'] for r in row['cases']}==set(times)
   gain=math.exp(sum(math.log(times[r['case']]/r['median_ms']) for r in row['cases'])/len(times))
   if gain>1.08:candidates.append((gain,row))
  for gain,row in sorted(candidates,key=lambda r:-r[0])[:3]:
   origin=next(r for r in read(batch/'knowledge/initial-variants.json') if r['name']==row['name'])
   source=Path(origin['path']);assert hashlib.sha256(source.read_bytes()).hexdigest()==row['source_sha256']
   out=batch/'results/upsample_bilinear2d/codex-initial'/row['source_sha256'];out.mkdir(parents=True,exist_ok=False)
   target=out/'main.py';target.write_bytes(source.read_bytes())
   result=campaign.step('validate:upsample_bilinear2d:codex-initial:'+row['source_sha256'][:12],lambda:campaign.candidate('upsample_bilinear2d',target,out,'codex-'+row['name']))
   if result and result.get('status')=='passed':
    operator=campaign.state['operators'].setdefault('upsample_bilinear2d',dict(status='queued'))
    old=operator.get('best')
    if not old or result['latency_ms']<old['latency_ms']:operator['best']=result;campaign.save()
  results=read(batch/'research/reference-stability/results.json')
  unstable=[r for r in results['rows'] if r['mode']=='default' and r.get('cumulative_trapezoid',{}).get('empty_intersections',0)>0]
  deterministic=[r for r in results['rows'] if r['mode']=='torch_deterministic']
  knowledge=dict(reference_probe=str(batch/'research/reference-stability/results.json'),unstable_default=unstable,
                 deterministic_experiments=deterministic,policy='原验收保持默认；确定性选项仅隔离诊断，不将新设置结果混作原合同达标')
  save(batch/'knowledge/reference-new-findings.json',knowledge)
  if unstable:
   for number in [1,2]:
    campaign.step('search:cumulative_trapezoid:'+str(number),lambda:dict(status='reference_unstable_search_deferred',reason='原参考同输入容差不自洽；已完成最小输入及官方确定性隔离诊断，当前不消耗模型预算追偶然通过',evidence=knowledge['reference_probe']))
  state.update(status='initial_diagnostics_completed',current_job='continuous_kg_search');save(path,state)
  try:
   campaign.execute()
   state.update(status='completed',current_job=None);save(path,state)
  except Exception as error:
   campaign.state.update(status='needs_attention',error=type(error).__name__+': '+str(error));campaign.save()
   state.update(status='needs_attention',error=type(error).__name__+': '+str(error));save(path,state);raise

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True)
 execute(p.parse_args().batch)
