"""原worker和控制器均退出后，保留并验证CC最后源码，再恢复原串行队列。"""
from pathlib import Path
import datetime,fcntl,hashlib,importlib.util,json,shlex,subprocess,urllib.request
root=Path('/data/hanle/ascend-optimization')
repo=root/'Ascend-Optimization';project=repo/'project/Ascend-Optimization-v1'
batch=project/'operators/campaigns/remaining-five-20261023-v2';tools=project/'tools/profiling/hgkda'
workspace=root/'runs/remaining-five-20261023-v2/kg-runs/fractional_max_pool3d-remaining-five-20261023-v2-attempt-1'
worker=json.loads((workspace/'.kernelgen/run-process.json').read_text())
assert worker.get('finished_at') and worker['state']!='RUNNING','原worker仍未退出，不接管设备'
state=json.loads((batch/'endurance-state.json').read_text())
assert state['status']=='needs_attention',state['status']
session='hanle-remaining-five-20261023-v2'
assert subprocess.run(['tmux','has-session','-t',session],stdout=subprocess.PIPE,stderr=subprocess.PIPE).returncode!=0,'原控制器仍在运行'
source=batch/'research/fractional_max_pool3d/captured-native/ee9b00f476ff20c79c0adb7262adcd85e775613ff19682a520ec2728d88e77e6/main.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='ee9b00f476ff20c79c0adb7262adcd85e775613ff19682a520ec2728d88e77e6'
assert hashlib.sha256((workspace/'stages/optimize/work/tmp/main.py').read_bytes()).hexdigest()==hashlib.sha256(source.read_bytes()).hexdigest(),'最终工作区另有新修改，应先保存后审查'
spec=importlib.util.spec_from_file_location('remaining_captured_resume',tools/'run-remaining-five.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
with (root/'runs/physical-card-7-campaign.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 campaign=module.Campaign(batch.name);campaign.q.idle()
 folder=batch/'results/fractional_max_pool3d/captured-native'/hashlib.sha256(source.read_bytes()).hexdigest()
 folder.mkdir(parents=True,exist_ok=True)
 def special():
  target=folder/'special.json'
  if not target.exists():
   try:campaign.run('captured-fractional-special',[module.base.PY,tools/'check-remaining-special.py','--operator','fractional_max_pool3d','--source',source,'--output',target],1800)
   except RuntimeError:
    if not target.exists():raise
  data=json.loads(target.read_text())
  assert data['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
  return dict(status=data['status'],passed=data['passed'],total=data['total'],check=str(target))
 special_result=campaign.step('captured-special:fractional_max_pool3d:ee9b00f476ff',special)
 print('SPECIAL',json.dumps(special_result),flush=True)
 if special_result and special_result['status']=='passed':
  result=campaign.step('validate:fractional_max_pool3d:captured-native:ee9b00f476ff',lambda:campaign.candidate('fractional_max_pool3d',source,folder,'captured-native'))
  if result and result['status']=='passed':
   previous=campaign.state['operators']['fractional_max_pool3d'].get('best')
   if not previous or result['latency_ms']<previous['latency_ms']:
    campaign.state['operators']['fractional_max_pool3d']['best']=result
   campaign.save()
  print('CANDIDATE',json.dumps(result),flush=True)
 review=batch/'fractional-captured-native-recovery.json'
 assert not review.exists()
 review.write_text(json.dumps(dict(status='validated_for_resume',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    contribution='CC/DeepSeek首轮最后源码；Codex保存源码、追加独立门禁/正式验证并恢复队列',
    native_worker=worker,special=special_result,previous_controller_error=state.get('error'),
    time=datetime.datetime.utcnow().isoformat()+'Z'),ensure_ascii=False,indent=2)+'\n')
# 卡锁已释放，原worker退出，恢复同一冻结工具/合同队列；不覆盖旧日志。
runtime=root/'runs/remaining-five-20261023-v2'
script=runtime/'launch.sh';assert script.is_file()
subprocess.run(['tmux','new-session','-d','-s',session,'bash '+shlex.quote(str(script))+' >>'+shlex.quote(str(runtime/'controller.log'))+' 2>&1'],check=True)
print('RESUMED',session,flush=True)
