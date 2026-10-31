"""三路对照与优化队列：卡锁串行、固定源码复验、任务边界处理新候选。"""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
NEXT=HERE.parent/'next_phase'
sys.path.insert(0,str(NEXT))
# 本文件名也叫run，显式加载原模块，避免导入自身。
import importlib.util
spec=importlib.util.spec_from_file_location('next_phase_runner',NEXT/'run.py')
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
sys.modules['run']=base
from proposals import process


class Runner(base.Runner):
    def disqualification(self,result):
        path=self.batch/'disqualified-sources.json'
        if not path.exists() or not result:return None
        return json.loads(path.read_text(encoding='utf8')).get(result.get('source_sha256'))

    def reconcile_selection(self):
        from selection_roles import optimization_jobs
        all_jobs=self.state['jobs']
        for row in self.state.get('operators',{}).values():
            if self.disqualification(row.get('best')):
                base.retain_unqualified(row,row['best'],'source_review_rejected')
                row.pop('best',None)
        self.state['jobs']={key:job for key,job in optimization_jobs(all_jobs).items()
                            if not self.disqualification(job.get('result'))}
        try:
            super().reconcile_selection()
        finally:
            self.state['jobs']=all_jobs

    def evaluate(self,op,source,out,label):
        manifest=source.parent/'proposal.json'
        spec=json.loads(manifest.read_text(encoding='utf8')) if manifest.exists() else {}
        if spec.get('implementation_language')!='python':
            return super().evaluate(op,source,out,label)
        identity=source.parent/spec['native_manifest']
        frozen=json.loads(identity.read_text(encoding='utf8'))
        import hashlib
        for path,sha in frozen['files'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:
                raise self.module.Attention('原生候选附件身份不同 '+path)
        self.q.run('evaluate-native-'+op+'-'+label,base.quote([base.PY,HERE/'evaluate_reference.py',
            '--operator',op,'--source',source,'--leaf',self.batch/self.q.records[op]['catalog_leaf'],
            '--output',out,'--server',self.q.server,'--label',label,'--kind',spec['backend']]),2100)
        path=out/(label+'-1.result.json');value=json.loads(path.read_text(encoding='utf8'))
        return dict(status=value.get('status'),geo_mean=value.get('geo_mean'),num_passed=value.get('num_passed'),
            num_workloads=value.get('num_workloads'),result=str(path),source=str(source),source_sha256=spec['source_sha256'],
            backend=spec['backend'],native_manifest=str(identity),attribution=spec['attribution'],
            latency_ms={x['uuid']:x['latency_ms'] for x in value.get('per_workload',[]) if x.get('phase')=='timing' and x.get('latency_ms') is not None})

    def consider_best(self,row,result):
        if self.disqualification(result):return
        if result and result.get('extra_checks'):
            result['supplementary_status']=base.extra_qualification(result)
        base.Runner.consider_best(row,result)

    def search_source(self,op,attempt):
        best=self.state.get('operators',{}).get(op,{}).get('best') or {}
        if 'AscendC' in best.get('backend',''):
            # 原KG搜索保持Triton种子，原生候选另走明确的Python/Ascend C合同入口。
            return self.starting_source(op)
        return super().search_source(op,attempt)

    def save(self):
        base.save_json(self.path,self.state)
        lines=['# 三路对照与优化进展','','本批完整KGS调用时间；历史与当前结果分别保存。','',
               '|算子|本批HG-KDA|完整评测|补充检查|阶段|','|---|---:|---|---|---|']
        for op in self.q.records:
            row=self.state.get('operators',{}).get(op,{})
            best=row.get('best') or {}
            score=('%.3f×'%best['geo_mean']) if isinstance(best.get('geo_mean'),(int,float)) else '—'
            lines.append('|'+op+'|'+score+'|'+str(best.get('num_passed','—'))+'/'+str(best.get('num_workloads','—'))+'|'+best.get('supplementary_status','待检查')+'|'+row.get('status','待运行')+'|')
        (self.batch/'进展.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

    def verify(self,op):
        source=self.starting_source(op)
        out=self.batch/'results'/op/'current-recheck'
        result=self.job('verify-current:'+op,lambda:self.evaluate(op,source,out/'formal','HG-KDA-current'))
        row=self.state['operators'].setdefault(op,{})
        if not result or result.get('status')!='PASSED':
            row['status']='本批完整正确性未通过';self.save();return
        gates=out/'extra-checks.json'
        qualification=self.job('verify-extra:'+op,lambda:self.supplementary(op,source,gates,'verify-extra-'+op))
        result.update(extra_checks=str(gates),supplementary_status=qualification,attribution='历史有效源码本批原样复验；不计新优化')
        if qualification=='passed':
            self.consider_best(row,result)
            row['status']='本批复验达标' if result.get('geo_mean',0)>.8 else '本批正确，性能待优化'
        else:
            row['status']='本批补充检查未通过'
        self.save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',default='threeway-20261030')
    p.add_argument('--phase',choices=['verify','references','search','proposals','finalize','all'],default='verify')
    p.add_argument('--operators',nargs='*')
    a=p.parse_args()
    with (base.ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        marker=base.ROOT/'runs'/a.batch/'STOP-AFTER-JOB'
        if marker.exists():
            if not marker.read_text().startswith('Controller handoff at own job boundary:'):
                raise RuntimeError('存在非本次控制器交接的停止标记，保留现场')
            marker.rename(marker.with_name('STOP-AFTER-JOB.handoff-'+str(time.time_ns())))
        r=Runner(a.batch)
        r.state['active_started_epoch']=time.time()
        r.save()
        try:
            if a.phase in {'verify','all'}:
                order=list(dict.fromkeys(r.q.plan['priority_operators']+r.q.plan['verification_operators']))
                for op in a.operators or order:
                    r.verify(op)
            if a.phase in {'references','all'}:
                if a.phase=='all':process(r)
                from references import process as process_references
                process_references(r)
            if a.phase=='all':
                process(r)
            if a.phase in {'search','all'}:
                for op in a.operators or r.q.plan['priority_operators']:
                    best=r.state['operators'].get(op,{}).get('best') or {}
                    if base.extra_qualification(best)=='passed' and best.get('geo_mean',0)>.8:
                        continue
                    attempt=r.state.get('search_recovery_attempts',{}).get(op,1)
                    result=r.job('search:'+op+':'+str(attempt),lambda op=op,attempt=attempt:r.search(op,attempt))
                    row=r.state['operators'].setdefault(op,{})
                    r.consider_best(row,result)
                    r.save()
            if a.phase in {'proposals','all'}:
                process(r)
            if a.phase in {'all','finalize'}:
                from finalize import process as finalize
                finalize(r)
            r.wait_idle()
            r.state.update(status=a.phase+'_completed',current_job=None,finished_at=r.module.now())
            r.save()
        except Exception as error:
            r.state.update(status='needs_attention',error=type(error).__name__+': '+str(error))
            r.save()
            raise


if __name__=='__main__':
    main()
