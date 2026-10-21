#!/usr/bin/env python3
"""从控制端只读查看夜间队列、KG阶段与服务状态。"""
import argparse
import importlib.util
import json
from pathlib import Path
import re

spec=importlib.util.spec_from_file_location('control',Path(__file__).with_name('control.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
REMOTE=r'''
import json,subprocess,urllib.request
from pathlib import Path
root=Path(parameters['root']);batch=parameters['batch']
directory=root/'Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns'/batch
runtime=root/'runs'/batch
out={'batch':batch,'state':json.loads((directory/'queue-state.json').read_text()) if (directory/'queue-state.json').exists() else None,
 'sessions':subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True).stdout.splitlines(),
 'workers':[],'services':[],
 'finalization':json.loads((directory/'finalization.json').read_text()) if (directory/'finalization.json').exists() else None}
for name in ['postflight','repair-verification','delivery-finalization','completion-audit']:
 path=directory/(name+'.json')
 out[name]=json.loads(path.read_text()) if path.exists() else None
audit=subprocess.run(['docker','exec','tle_yy','ps','-eo','pid,etimes,pcpu,args'],capture_output=True,text=True)
processes={}
if audit.returncode==0:
 for line in audit.stdout.splitlines()[1:]:
  fields=line.split(None,3)
  if len(fields)==4:processes[int(fields[0])]={'elapsed_seconds':int(fields[1]),'cpu_percent':float(fields[2]),'args':fields[3]}
else:out['worker_process_audit_error']='容器ps检查失败，不能仅凭状态文件认定worker存活'
out['queue_processes']=[]
host_audit=subprocess.run(['ps','-eo','pid,etimes,args'],capture_output=True,text=True)
host_processes={}
for line in host_audit.stdout.splitlines()[1:]:
 fields=line.split(None,2)
 if len(fields)!=3:continue
 arguments=fields[2];parts=arguments.split()
 if not parts or not Path(parts[0]).name.startswith('python'):continue
 host_processes[int(fields[0])]={'elapsed_seconds':int(fields[1]),'parts':parts}
 for name in ['run-nightly.py','run-semantic-repair.py','run-selu-repair.py','postflight-nightly.py']:
  script=str(root/'Ascend-Optimization/project/Ascend-Optimization-v1/tools/profiling/hgkda'/name)
  if script not in parts:continue
  if '--batch' not in parts:continue
  index=parts.index('--batch')+1
  if index<len(parts) and parts[index]==batch:
   out['queue_processes'].append({'script':name,'pid':int(fields[0]),'elapsed_seconds':int(fields[1])})
out['related_batches']=[]
for plan_path in sorted(directory.parent.glob(batch+'-*/plan.json')):
 path=plan_path.with_name('queue-state.json')
 state=json.loads(path.read_text()) if path.exists() else {}
 related=path.parent.name;matches=[]
 for pid,process in host_processes.items():
  parts=process['parts']
  if '--batch' not in parts:continue
  index=parts.index('--batch')+1
  if index>=len(parts) or parts[index]!=related:continue
  if not any(str(root/'Ascend-Optimization/project/Ascend-Optimization-v1/tools/profiling/hgkda'/name) in parts
             for name in ['run-nightly.py','run-semantic-repair.py','run-selu-repair.py']):continue
  matches.append({'pid':pid,'elapsed_seconds':process['elapsed_seconds']})
 out['related_batches'].append({'batch':related,'status':state.get('status','queued' if matches else 'not_started'),
  'current_job':state.get('current_job'),'error':state.get('error'),
  'process_audit_available':host_audit.returncode==0,'live_processes':matches,
  'scores':state.get('result',{}).get('scores',[])})
audit_path=directory/'selu-launch-audit-state.json'
if audit_path.exists():
 state=json.loads(audit_path.read_text());process=host_processes.get(state.get('pid'))
 out['selu_launch_audit']={**state,'process_live':bool(process and
  str(runtime/'selu-launch-audit-controller.py') in process['parts']) if host_audit.returncode==0 else None}
out['reference_execution_audit']=json.loads((directory/'reference-execution-audit.json').read_text()) if (directory/'reference-execution-audit.json').exists() else {}
if out['state'] is None and (runtime/'queue.log').exists():
 out['startup_log_tail']=(runtime/'queue.log').read_text(errors='replace')[-2500:]
for path in sorted((runtime/'kg-runs').glob('*/.kernelgen/run-progress.json')):
 data=json.loads(path.read_text());work=path.parents[1]
 row={'workspace':work.name,**{k:data.get(k) for k in ['state','stage','updated_at']}}
 metadata=path.with_name('run-process.json')
 if metadata.exists():
  identity=json.loads(metadata.read_text());row['pid']=identity.get('pid')
  process=processes.get(row['pid'])
  row['worker_live']=(bool(process and 'kernelgen.cli.worker' in process['args'] and str(work) in process['args']) if audit.returncode==0 else None)
  if row['worker_live']:row.update({k:process[k] for k in ['elapsed_seconds','cpu_percent']})
 else:row['worker_live']=None
 out['workers'].append(row)
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for port in [19655,19657,19659]:
 try:
  data=json.load(opener.open('http://127.0.0.1:%d/status'%port,timeout=10))
  out['services'].append({'port':port,'scheduler':data['scheduler']})
 except Exception as error:out['services'].append({'port':port,'error':str(error)})
print(json.dumps(out,ensure_ascii=False))
'''


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='baai-910b')
    p.add_argument('--root',default='/data/hanle/ascend-optimization')
    p.add_argument('--batch',default='night-20261021-v3')
    p.add_argument('--json',action='store_true')
    a=p.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]+',a.batch):p.error('非法批次名')
    data=json.loads(c.remote_python(a.host,{'root':a.root,'batch':a.batch},REMOTE))
    if a.json:
        print(json.dumps(data,ensure_ascii=False,indent=2))
    else:
        state=data['state'] or {}
        print('批次:',a.batch,'状态:',state.get('status','尚未启动'),'当前:',state.get('current_job'))
        if state.get('error'):print('队列提示:',state['error'])
        if data.get('startup_log_tail'):print('启动日志:',data['startup_log_tail'])
        for name,item in state.get('jobs',{}).items():
            op=name.split(':')[1]
            fallback=data.get('reference_execution_audit',{}).get(op,{}).get('cpu_fallback')
            print(name,item['status'],'参考包含CPU回退，原始计时单独保存' if fallback else '/'.join('%.3f×'%x for x in item.get('scores',[])))
        for item in data['workers']:
            live={True:'存活',False:'已退出',None:'未核实'}[item.get('worker_live')]
            print('KG:',item['workspace'],'记录='+str(item['state']),item['stage'],'进程='+live,'pid='+str(item.get('pid')))
        for item in data.get('queue_processes',[]):
            print('控制进程:',item['script'],'pid='+str(item['pid']),'已运行秒数='+str(item['elapsed_seconds']))
        for item in data['services']:
            s=item.get('scheduler',{})
            print('KGS:',item['port'],'healthy='+str(s.get('healthy')),'active='+str(s.get('active')))
        if data.get('finalization'):print('原始组严格核验:',data['finalization']['status'])
        if data.get('delivery-finalization'):print('交付汇总:',data['delivery-finalization']['status'])
        if data.get('completion-audit'):print('实验交付核验:',data['completion-audit']['status'])
        if data.get('postflight'):print('后置汇总:',data['postflight']['status'])
        for item in data.get('related_batches',[]):
            live=('存活' if item['live_processes'] else '未发现运行进程') if item['process_audit_available'] else '未核实'
            print('附加批次:',item['batch'],item['status'],'进程='+live,
                  '/'.join('%.3f×'%x for x in item.get('scores',[])))
            if item.get('error'):print('附加提示:',item['error'])
        if data.get('selu_launch_audit'):
            item=data['selu_launch_audit']
            print('SELU启动路径审计:',item['status'],'进程='+str(item['process_live']))
        if data.get('worker_process_audit_error'):print('进程检查提示:',data['worker_process_audit_error'])
