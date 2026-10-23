"""仅在所有前序队列退出后，验证宿主类型化输出指针的双元素打包存储。"""
from pathlib import Path
import datetime,fcntl,hashlib,importlib.util,json,math,shlex,shutil,time,urllib.request

ROOT=Path('/data/hanle/ascend-optimization')
PROJECT=ROOT/'Ascend-Optimization/project/Ascend-Optimization-v1'
BATCH=PROJECT/'operators/campaigns/remaining-five-20261023-v2'
TOOLS=PROJECT/'tools/profiling/hgkda'
PY='/usr/local/python3.11.15/bin/python3.11'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def execute():
 state_path=BATCH/'packed-v2-state.json';assert not state_path.exists()
 state=dict(status='waiting_for_finalization',started_at=datetime.datetime.utcnow().isoformat()+'Z');save(state_path,state)
 end=time.monotonic()+1200
 while read(BATCH/'finalization-state.json')['status']!='completed':
  assert read(BATCH/'finalization-state.json')['status']!='needs_attention'
  if time.monotonic()>end:raise TimeoutError('前序收尾尚未结束；未使用设备')
  time.sleep(10)
 with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  module=load('packed_v2_queue',TOOLS/'run-nightly.py');q=module.Queue(BATCH.name,ROOT,'tle_yy')
  base=load('packed_v2_campaign',TOOLS/'run-remaining-five.py')
  campaign=object.__new__(base.Campaign);campaign.batch=BATCH;campaign.q=q
  def run(key,parts,budget=1800):
   q.idle();state.update(status='running',current_job=key);save(state_path,state)
   q.run('packed-v2-'+key,'export PYTHONDONTWRITEBYTECODE=1\n'+' '.join(shlex.quote(str(x)) for x in parts),budget)
  campaign.run=run
  out=BATCH/'research/upsample_bilinear2d/packed-store-v2-probe';out.mkdir(parents=True,exist_ok=False)
  shutil.copyfile(Path(__file__),out/'reproduce.py')
  try:
   source=(BATCH/'knowledge/upsample-packed-store-v2-probe.py').read_text()
   (out/'probe.py').write_text(source)
   request=dict(command=['{python}','probe.py'],files=[dict(path='probe.py',content=source)],timeout_seconds=300)
   save(out/'request.json',request)
   q.idle();state.update(status='running',current_job='probe');save(state_path,state)
   response=json.load(q.opener.open(urllib.request.Request(q.server+'/debug/jobs',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'},method='POST'),timeout=330))
   save(out/'response.json',response);state['job_id']=response['job_id']
   if response['status']!='SUCCEEDED' or response.get('exit_code')!=0:
    state.update(status='completed',outcome='probe_failed');save(state_path,state);return
   data=read(out/'timings.json');source=BATCH/'knowledge/upsample-packed-store-v2.py'
   assert data['source_sha256']==sha(source)
   ratios=[]
   for case in sorted({r['case'] for r in data['rows']}):
    rows=[r for r in data['rows'] if r['case']==case];assert len(rows)==2 and all(r['correct'] for r in rows)
    values={r['variant']:r['walltime_ms'] for r in rows}
    assert len(values)==2 and all(v>0 for v in values.values())
    ratios.append(values['parent']/values['packed_store'])
   assert len(ratios)==2
   gain=math.exp(sum(math.log(v) for v in ratios)/len(ratios));state['diagnostic_gain']=gain;save(state_path,state)
   if gain<=1.05:
    state.update(status='completed',outcome='no_diagnostic_gain');save(state_path,state);return
   folder=BATCH/'results/upsample_bilinear2d/codex-packed-v2/candidates'/sha(source)
   folder.mkdir(parents=True,exist_ok=False);shutil.copyfile(source,folder/'main.py')
   candidate=campaign.candidate('upsample_bilinear2d',folder/'main.py',folder,'codex-packed-v2')
   state['candidate']=candidate;save(state_path,state)
   if candidate['status']!='passed':
    state.update(status='completed',outcome='candidate_rejected');save(state_path,state);return
   special=folder/'special.json'
   try:run('special',[PY,TOOLS/'check-remaining-special.py','--operator','upsample_bilinear2d','--source',folder/'main.py','--output',special],1800)
   except RuntimeError:
    if not special.exists():raise
   extra=read(special);candidate['special']=str(special)
   manifest=read(BATCH/'final-delivery.json');old=next(r for r in manifest['records'] if r['operator']=='upsample_bilinear2d')
   if extra['status']!='passed' or candidate['latency_ms']>=old['selected']['latency_ms']:
    state.update(status='completed',outcome='special_failed_or_no_formal_gain');save(state_path,state);return
   assert extra['source_sha256']==candidate['source_sha256'] and extra['passed']==extra['total']==36
   audit=load('packed_v2_formal_audit',TOOLS/'audit-endurance.py')
   record=next(r for r in read(BATCH/'plan.json')['records'] if r['operator']=='upsample_bilinear2d')
   result=read(Path(candidate['result']));audit.verify_formal_contract(record,Path(candidate['result']).parent,result)
   req=read(Path(candidate['result']).with_name(Path(candidate['result']).name.replace('.result.json','.request.json')))
   assert len(req['implementation']['sources'])==1 and req['implementation']['sources'][0]['content']==source.read_text()
   target=BATCH/'deliverables/reviewed/upsample_bilinear2d/packed-v2.py';assert not target.exists();shutil.copyfile(source,target)
   facts=BATCH/'research/upsample_bilinear2d/packed-v2-final'
   campaign.observe('upsample_bilinear2d',target,facts);candidate['facts']=str(facts/'facts.json')
   try:campaign.metrics('upsample_bilinear2d',candidate,BATCH/'research/upsample_bilinear2d/packed-v2-final-metrics')
   except RuntimeError as error:q.idle();state['metrics_error']=str(error)
   before=BATCH/'final-delivery-before-packed-v2.json';assert not before.exists();shutil.copyfile(BATCH/'final-delivery.json',before)
   old.update(source=str(target),selected=candidate,score=result['geo_mean'],gates=candidate['gates']+[str(special)],meets_numerical_speed_target=result['geo_mean']>.8)
   manifest['under_target']=[r['operator'] for r in manifest['records'] if r['score']<=.8]
   save(BATCH/'final-delivery.json',manifest)
   state.update(status='completed',outcome='selected',score=result['geo_mean']);save(state_path,state)
  except Exception as error:
   state.update(status='needs_attention',error=type(error).__name__+': '+str(error));save(state_path,state);raise


if __name__=='__main__':execute()
