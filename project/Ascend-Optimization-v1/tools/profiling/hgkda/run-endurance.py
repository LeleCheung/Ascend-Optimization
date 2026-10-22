#!/usr/bin/env python3
"""宿主tmux连续队列：六算子、有限自适应搜索、独立门禁、真实路径和最佳交付。"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request

TOOLS=Path(__file__).resolve().parent
ROOT=Path('/data/hanle/ascend-optimization')
PY='/usr/local/python3.11.15/bin/python3.11'


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def quote(parts):return ' '.join(shlex.quote(str(p)) for p in parts)
def read(p):return json.loads(p.read_text())


def latency(result):
    rows=[r for r in result.get('per_workload',[]) if r.get('phase')=='timing']
    if not rows or any(not r.get('latency_ms') or r['latency_ms']<=0 for r in rows):
        raise ValueError('缺少有效逐case实际延迟')
    return math.exp(sum(math.log(r['latency_ms']) for r in rows)/len(rows))


def improved(parent,candidate):
    a={r['uuid']:r['latency_ms'] for r in parent['per_workload'] if r.get('phase')=='timing'}
    b={r['uuid']:r['latency_ms'] for r in candidate['per_workload'] if r.get('phase')=='timing'}
    if set(a)!=set(b) or not a:raise ValueError('比较的case集合不同')
    return math.exp(sum(math.log(a[k]/b[k]) for k in a)/len(a))


class Campaign:
    def __init__(self,batch):
        self.module=load('endurance_queue',TOOLS/'run-nightly.py')
        self.q=self.module.Queue(batch,ROOT,'tle_yy')
        self.batch=self.q.batch;self.path=self.batch/'endurance-state.json'
        self.state=read(self.path) if self.path.exists() else dict(status='created',operators={},jobs={},created_at=self.module.now())
        for name,digest in read(self.batch/'knowledge/tool-sha256.json').items():
            if sha(TOOLS/name)!=digest:raise self.module.Attention('冻结控制器身份改变: '+name)
        self.state.setdefault('started_epoch',time.time())
        self.save()
        self.deadline=time.monotonic()+max(0,self.q.plan['duration_hours']*3600-(time.time()-self.state['started_epoch']))

    def save(self):
        self.state['updated_at']=self.module.now();self.module.save(self.path,self.state)

    def step(self,key,fn):
        job=self.state['jobs'].get(key,{})
        if job.get('status') in ['completed','failed']:return job.get('result')
        self.q.idle()
        self.state.update(status='running',current_job=key)
        self.state['jobs'][key]=dict(status='running',started_at=self.module.now());self.save()
        try:
            value=fn()
            self.state['jobs'][key].update(status='completed',result=value,finished_at=self.module.now())
        except self.module.Attention:raise
        except Exception as error:
            # 只有服务已确认空闲才可继续下一项；未知设备操作交由控制端处理，不盲目重提。
            self.q.idle()
            value=None
            self.state['jobs'][key].update(status='failed',error=type(error).__name__+': '+str(error),finished_at=self.module.now())
        self.save();self.report();return value

    def run(self,key,parts,budget=1800):
        self.q.run(key,quote(parts),budget)

    def gates(self,op,source,out):
        checks=([('semantics','check-pointwise-semantics.py'),('dense','check-dense-pointwise.py'),
                 ('ranges','check-pointwise-ranges.py')] if op=='selu_' else [('semantics','check-endurance.py')])
        for name,script in checks:
            target=out/(name+'.json')
            if not target.exists():
                try:self.run('gate-'+op+'-'+out.name+'-'+name,[PY,TOOLS/script,'--operator',op,'--source',source,'--output',target])
                except RuntimeError:
                    if not target.exists():raise
            d=read(target)
            if d['source_sha256']!=sha(source):raise ValueError('门禁源码身份改变')
            if d['status'].lower()!='passed':return dict(status='semantic_failed',check=str(target),passed=d.get('passed'),total=d.get('total'))
        return dict(status='passed',checks=[str(out/(n+'.json')) for n,s in checks])

    def evaluate(self,op,source,out,label):
        digest=sha(source)
        # 一批同字节同合同只正式测一次，失败也不重提。
        for old in self.batch.glob('results/'+op+'/**/formal/*.provenance.json'):
            meta=read(old)
            if meta.get('source_sha256')!=digest:continue
            result=old.with_name(old.name.replace('.provenance.json','-1.result.json'))
            if result.exists():
                d=read(result)
                if d.get('status')=='PASSED' and d.get('num_passed')==d.get('num_workloads'):
                    return dict(status='passed',source=str(source),source_sha256=digest,result=str(result),geo_mean=d['geo_mean'],latency_ms=latency(d),reused=True)
                return dict(status='formal_failed',source_sha256=digest,result=str(result))
        formal=out/'formal'
        try:self.q.evaluate(op,source,formal,label,1)
        except (RuntimeError,AssertionError):
            result=formal/(label+'-1.result.json')
            if not result.exists():raise
        result=formal/(label+'-1.result.json');d=read(result)
        if d.get('status')!='PASSED' or d.get('num_passed')!=d.get('num_workloads'):
            return dict(status='formal_failed',source_sha256=digest,result=str(result))
        return dict(status='passed',source=str(source),source_sha256=digest,result=str(result),
            geo_mean=d['geo_mean'],latency_ms=latency(d))

    def candidate(self,op,source,out,label):
        out.mkdir(parents=True,exist_ok=True)
        gate=self.gates(op,source,out)
        if gate['status']!='passed':return gate
        result=self.evaluate(op,source,out,label);result['gates']=gate['checks']
        return result

    def baseline(self,op):
        record=self.q.records[op];source=self.batch/record['seed'];out=self.batch/'results'/op/'parent'
        out.mkdir(parents=True,exist_ok=True)
        # 父源码可能有已知精度问题，仍保留本批正式失败及补充失败，为模型修复提供证据。
        gate=self.gates(op,source,out)
        result=self.evaluate(op,source,out,'parent');result['gate_result']=gate
        if gate['status']!='passed' and result['status']=='passed':result['status']='semantic_failed'
        elif gate['status']=='passed':result['gates']=gate['checks']
        return result

    def observe(self,op,source,out):
        leaf=self.batch/self.q.records[op]['catalog_leaf']
        self.run('observe-'+op+'-'+out.name,[PY,TOOLS/'observe-endurance.py','--operator',op,
            '--source',source,'--leaf',leaf,'--output',out,'--index','0'],1200)
        return dict(path=str(out/'facts.json'),status=read(out/'facts.json')['status'])

    def metrics(self,op,selected,out):
        result=read(Path(selected['result']))
        rows=[r for r in result['per_workload'] if r.get('phase')=='timing']
        # 取候选累计耗时最高的代表，不从其他算子猜测case名。
        row=max(rows,key=lambda r:r['latency_ms'])
        formal=Path(selected['result']).parent
        request=Path(selected['result']).with_name(Path(selected['result']).name.replace('.result.json','.request.json'))
        self.run('metrics-'+op+'-'+out.name,[PY,self.q.project/'tools/evaluation/profile-operator.py',
            '--evaluation',selected['result'],'--request',request,'--inspect',formal/'inspect.json',
            '--case-id',row['uuid'],'--output',out,'--server',self.q.server,
            '--level','metrics','--aic-metrics','PipeUtilization','--timeout','240'],420)
        return dict(path=str(out/'response.json'),status=read(out/'response.json')['status'])

    def native(self,op,attempt,seed,overlay,budget):
        suffix=self.batch.name+'-'+attempt
        workspace=self.q.runtime/'kg-runs'/(op+'-'+suffix)
        if workspace.exists():
            metadata=workspace/'.kernelgen/run-process.json'
            if metadata.exists() and read(metadata).get('finished_at'):return workspace
            raise self.module.Attention('已有worker未确认结束；不重复启动: '+str(workspace))
        exports=dict(KG_SOURCE_ROOT=str(overlay),KG_RUN_ROOT=str(self.q.runtime/'kg-runs'),
            KG_CATALOG_PATH=str(self.batch/'catalog'),KG_WALLTIME_PORT='19659',KG_PROFILE_PORT='19659',KG_MODEL='deepseek-flash')
        body='\n'.join('export '+k+'='+shlex.quote(v) for k,v in exports.items())+'\n'+quote([
            'bash',self.q.project/'tools/evaluation/run-kg-operator-910b.sh',op,'profile',suffix,
            self.q.plan['passes'][int(attempt[-1])-1]['rounds'],seed])
        command='set -euo pipefail\nsource /usr/local/Ascend/cann-9.0.0/set_env.sh\nexport ASCEND_RT_VISIBLE_DEVICES=7\n'
        command+='export PYTHONPATH='+shlex.quote(':'.join(str(p) for p in [self.q.repo/'third_party/FlagGems/src',self.q.repo/'sources/kernelgen_server/client',self.q.repo/'sources/kernelgen_server',self.q.repo/'sources']))+'\n'
        command+='export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost\nunset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy\n'+body
        log=self.q.runtime/(op+'-'+attempt+'.log')
        started=time.monotonic();seen={};cancelled=set();requested=None
        with log.open('ab') as stream:
            process=subprocess.Popen(['docker','exec','-i','tle_yy','bash','-s'],stdin=subprocess.PIPE,stdout=stream,stderr=subprocess.STDOUT)
            process.stdin.write(command.encode());process.stdin.close()
            while process.poll() is None:
                active=workspace/'.kernelgen/active-server-operations.json'
                if active.exists():
                    for item in read(active).get('operations',{}).values():
                        oid=item['operation_id'];seen.setdefault(oid,time.monotonic())
                        if (item['kind']=='profile' and item['server_url']==self.q.server and oid not in cancelled
                                and time.monotonic()-seen[oid]>self.q.plan['profile_budget_seconds']):
                            url=self.q.server+'/operations/'+oid
                            before=json.load(self.q.opener.open(url,timeout=15))
                            if before.get('kind')!='profile':raise self.module.Attention('操作身份改变')
                            after=json.load(self.q.opener.open(urllib.request.Request(url,method='DELETE'),timeout=20))
                            cancelled.add(oid)
                            self.module.save(self.batch/('watchdog-'+op+'-'+attempt+'-'+hashlib.sha256(oid.encode()).hexdigest()[:12]+'.json'),
                                dict(operation=item,before=before,after=after,reason='本批单次profile预算240秒；只取消本worker登记操作，不重启服务',time=self.module.now()))
                if time.monotonic()-started>budget and requested is None:
                    result=subprocess.run(['docker','exec','-e','PYTHONPATH='+str(overlay), 'tle_yy',
                        str(ROOT/'runtime/kg-controller/venv/bin/kg'),'cancel',str(workspace),'--reason','夜间单项搜索预算到期，保留已通过轮次并转下项'],
                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
                    self.module.save(self.batch/('budget-'+op+'-'+attempt+'.json'),dict(returncode=result.returncode,output=result.stdout.decode(errors='replace'),workspace=str(workspace)))
                    requested=time.monotonic()
                if requested and time.monotonic()-requested>180:
                    raise self.module.Attention('已取消本worker但尚未确认退出；保留进程，不启动其他设备请求')
                self.module.save(self.batch/'heartbeat.json',dict(status='running',operator=op,attempt=attempt,
                    workspace=str(workspace),elapsed_seconds=round(time.monotonic()-started),time=self.module.now()))
                time.sleep(10)
        metadata=workspace/'.kernelgen/run-process.json'
        if not metadata.exists() or not read(metadata).get('finished_at'):
            raise self.module.Attention('KG退出但完成元数据未知')
        self.q.idle();return workspace

    def search(self,op,number):
        attempt='attempt-'+str(number);out=self.batch/'results'/op/attempt;out.mkdir(parents=True,exist_ok=True)
        opstate=self.state['operators'][op];best=opstate.get('best')
        seed=Path(best['source']) if best else self.batch/self.q.records[op]['seed']
        overlay=ROOT/'runtime'/self.batch.name/(op+'-'+attempt)/'sources'
        if not overlay.exists():
            self.run('overlay-'+op+'-'+attempt,[PY,TOOLS/'build-kg-overlay.py','--output',overlay,'--native-context'],600)
            # 只修改本批新覆盖层；模型实际载入的角色随后原样归档。
            evidence=dict(operator=op,seed=str(seed),seed_sha256=sha(seed),
                current_operator_state=opstate,completed_jobs={k:v for k,v in self.state['jobs'].items() if op in k},
                frozen_knowledge=[str(p) for p in (self.batch/'knowledge').iterdir()])
            paired=self.batch/'research'/op/'parent/facts.json'
            if paired.exists():
                facts=read(paired)
                evidence['paired_execution']=dict(case=facts['case'],source_sha256=facts['source_sha256'],
                    sides={side:dict(status=v['status'],trace=v.get('trace'),device_events=v.get('device_events',[])[:32],
                        total_device_events=len(v.get('device_events',[])),error=v.get('error')) for side,v in facts['sides'].items()})
            metrics=self.batch/'research'/op/'parent-metrics/response.json'
            if metrics.exists():
                data=read(metrics)
                evidence['parent_metrics']=dict(status=data.get('status'),metrics=data.get('metrics'),original=str(metrics))
            for p in (self.batch/'knowledge').glob('*.md'):
                if p.name!='夜间任务.md':evidence[p.name]=p.read_text()[:12000]
            brief=(self.batch/'knowledge/夜间任务.md').read_text()
            appended='\n\n# 本批连续优化任务\n\n'+brief+'\n\n## 本项已知证据与失败\n\n'+json.dumps(evidence,ensure_ascii=False,indent=2)+'\n'
            for name in ['kernel-profile-analyzer','kernel-knowledge-profile-analyzer','kernel-coder','kernel-knowledge-coder','kernel-optimize']:
                p=overlay/'kernelgen/.kernelgen/agents'/(name+'.md')
                p.write_text(p.read_text()+appended)
            overlay_meta=read(overlay/'hgkda-overlay.json')
            overlay_meta['endurance_extension']=dict(operator=op,attempt=attempt,
                task_sha256=sha(self.batch/'knowledge/夜间任务.md'),evidence_sha256=hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest(),
                actual_roles={p.name:sha(p) for p in (overlay/'kernelgen/.kernelgen/agents').glob('*.md')})
            self.module.save(overlay/'hgkda-overlay.json',overlay_meta)
            self.module.save(out/'input-provenance.json',dict(seed=str(seed),source_sha256=sha(seed),
                task_sha256=sha(self.batch/'knowledge/夜间任务.md'),evidence=evidence,
                roles={p.name:sha(p) for p in (overlay/'kernelgen/.kernelgen/agents').glob('*.md')},
                contribution='Codex机制诊断及持续控制；CC/DeepSeek实际生成修改；原生KGS完整评测'))
        workspace=self.native(op,attempt,seed,overlay,self.q.plan['passes'][number-1]['budget_seconds'])
        archive=out/'native-evidence'
        if not archive.exists():self.run('archive-'+op+'-'+attempt,[PY,self.q.project/'tools/evaluation/archive-kg-evidence.py',workspace,archive],900)
        roles=[]
        for p in workspace.rglob('kernel*profile-analyzer.md'):
            if p.is_file() and '本批连续优化任务' in p.read_text(errors='replace'):
                target=out/('loaded-'+sha(p)[:12]+'.md');target.write_bytes(p.read_bytes())
                roles.append(dict(original=str(p),archived=str(target),sha256=sha(target)))
        self.module.save(out/'role-load-evidence.json',dict(roles=roles))
        pool=[]
        for directory in workspace.glob('stages/optimize/work/.kernelgen/evals/round-*'):
            if not (directory/'result.json').exists():continue
            d=read(directory/'result.json')
            if d.get('status')!='PASSED' or d.get('is_hack') or d.get('num_passed')!=d.get('num_workloads'):continue
            sources=read(directory/'solution.json').get('sources',[])
            if len(sources)!=1 or sources[0].get('content')!=(directory/'main.py').read_text():continue
            pool.append((latency(d),directory))
        candidates=[];seen=set()
        for _,directory in sorted(pool,key=lambda x:x[0]):
            source=directory/'main.py';digest=sha(source)
            if digest in seen:continue
            seen.add(digest)
            target=out/'candidates'/digest/'main.py';target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
            result=self.step('validate:'+op+':'+attempt+':'+digest[:12],
                lambda source=target,directory=directory:self.candidate(op,source,target.parent,directory.name))
            if result and result.get('status')=='passed':candidates.append(result)
        # 本批按同合同候选实测延迟选，参考波动不能改变排序。
        valid=([best] if best else [])+candidates
        winner=min(valid,key=lambda r:r['latency_ms']) if valid else None
        if winner:
            opstate['best']=winner
            parent=opstate.get('parent')
            if parent and parent.get('status')=='passed':opstate['actual_speedup_over_parent']=improved(read(Path(parent['result'])),read(Path(winner['result'])))
        opstate.setdefault('attempts',[]).append(dict(attempt=attempt,workspace=str(workspace),valid_candidates=len(candidates),roles=len(roles)))
        self.save()
        return dict(workspace=str(workspace),valid_candidates=len(candidates),best=winner)

    def report(self):
        lines=['# 夜间连续优化进展','','性能为本批KGS完整调用相对PyTorch；各新源码一次正式复验。正在运行的批次不是最终成果。','']
        for group,label in [('excel_meeting','会议算子'),('flaggems_sglang','FlagGems-SGLang')]:
            lines+=['## '+label,'','| 算子 | 同批起点 | 当前有效最佳 | 实际加速 | 状态 |','|---|---:|---:|---:|---|']
            for op,r in self.q.records.items():
                if r['source']!=group:continue
                d=self.state['operators'].get(op,{});parent=d.get('parent') or {};best=d.get('best') or {}
                fmt=lambda x:'%.3f×'%x if isinstance(x,(float,int)) else '—'
                lines.append('| '+op+' | '+fmt(parent.get('geo_mean'))+' | '+fmt(best.get('geo_mean'))+' | '+fmt(d.get('actual_speedup_over_parent'))+' | '+d.get('status','待执行')+' |')
            lines.append('')
        (self.batch/'进展.md').write_text('\n'.join(lines)+'\n')

    def finish(self):
        manifest=dict(schema_version='hgkda-endurance-delivery-v1',records=[],final_repeats=1)
        (self.batch/'deliverables').mkdir(exist_ok=True)
        for op,d in self.state['operators'].items():
            best=d.get('best')
            if not best:continue
            source=Path(best['source']);digest=sha(source);assert digest==best['source_sha256']
            target=self.batch/'deliverables'/op/digest/'main.py';target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(source.read_bytes())
            checks=[read(Path(p)) for p in best.get('gates',[])]
            assert checks and all(c['status'].lower()=='passed' and c['source_sha256']==digest for c in checks)
            formal=read(Path(best['result']));assert formal['status']=='PASSED' and formal['num_passed']==formal['num_workloads']
            manifest['records'].append(dict(operator=op,delivered=str(target),selected=best,
                actual_speedup_over_parent=d.get('actual_speedup_over_parent'),new_source=digest!=self.q.records[op]['source_sha256']))
        self.module.save(self.batch/'deliverables/manifest.json',manifest)
        self.state.update(status='completed',current_job=None,completed_at=self.module.now())
        self.save();self.report()

    def execute(self):
        self.q.idle()
        self.step('baseline-version-check',lambda:self.run('verify-baseline',[PY,TOOLS/'verify-baseline.py','--server',self.q.server,'--output',self.batch/'baseline-verified.json'],600))
        if self.state['jobs']['baseline-version-check']['status']!='completed':raise self.module.Attention('固定版本核验失败')
        for pass_index in range(1,len(self.q.plan['passes'])+1):
            for op in self.q.records:
                if time.monotonic()>self.deadline:
                    self.state['budget_exhausted']=True;break
                d=self.state['operators'].setdefault(op,dict(status='queued'))
                if pass_index==1:
                    parent=self.step('parent:'+op,lambda op=op:self.baseline(op));d['parent']=parent
                    if parent and parent.get('status')=='passed' and 'best' not in d:d['best']=parent
                    seed=self.batch/self.q.records[op]['seed']
                    self.step('observe-parent:'+op,lambda op=op,seed=seed:self.observe(op,seed,self.batch/'research'/op/'parent'))
                    if parent and parent.get('status')=='passed':
                        self.step('metrics-parent:'+op,lambda op=op,parent=parent:self.metrics(op,parent,self.batch/'research'/op/'parent-metrics'))
                elif d.get('best',{}).get('geo_mean',0)>=.8 and (op!='selu_' or d.get('actual_speedup_over_parent',1)>1.03):
                    continue
                d['status']='搜索中';self.save()
                self.step('search:'+op+':'+str(pass_index),lambda op=op,number=pass_index:self.search(op,number))
                d['status']='已完成本轮搜索';self.save()
            if self.state.get('budget_exhausted'):break
        for op,d in self.state['operators'].items():
            best=d.get('best')
            if best:
                self.step('observe-final:'+op,lambda op=op,best=best:self.observe(op,Path(best['source']),self.batch/'research'/op/'final'))
                self.step('metrics-final:'+op,lambda op=op,best=best:self.metrics(op,best,self.batch/'research'/op/'final-metrics'))
                d['status']='有效最佳已交付' if best['geo_mean']>=.8 else '有效最佳未达0.8×'
            else:d['status']='无通过全部门禁的交付；失败证据保留'
        self.finish()


if __name__=='__main__':
    import fcntl
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True)
    a=p.parse_args()
    campaign=Campaign(a.batch)
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:campaign.execute()
        except Exception as e:
            campaign.state.update(status='needs_attention',error=type(e).__name__+': '+str(e));campaign.save();raise
