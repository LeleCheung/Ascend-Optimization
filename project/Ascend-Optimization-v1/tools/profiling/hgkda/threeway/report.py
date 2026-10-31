"""核对本批源码/合同/结果，生成中文进展与交付索引；不提交设备任务。"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[4]


def read(path):return json.loads(path.read_text(encoding='utf8'))


def build(batch):
    state=read(batch/'next-state.json');plan=read(batch/'plan.json')
    files={};rows=[];contracts={};external={}
    denied=read(batch/'disqualified-sources.json') if (batch/'disqualified-sources.json').exists() else {}
    def include(path,expected=None):
        path=Path(path).resolve()
        raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
        if expected and expected!=sha:raise ValueError('身份不符 '+str(path))
        try:relative=path.relative_to(PROJECT).as_posix()
        except ValueError:
            archived=batch/'external-evidence'/sha/path.name
            archived.parent.mkdir(parents=True,exist_ok=True)
            if archived.exists() and archived.read_bytes()!=raw:raise ValueError('外部证据归档身份不同')
            archived.write_bytes(raw)
            relative=archived.relative_to(PROJECT).as_posix();external[str(path)]=relative
        files[relative]=dict(sha256=sha,bytes=len(raw));return relative
    for name in ['plan.json','bootstrap.json','software-lock.json','targets.json']:
        include(batch/name)
    for name in ['disqualified-sources.json','baseline-verified-final.json','catalog-preflight-resolution.json']:
        if (batch/name).exists():include(batch/name)
    if (batch/'catalog/manifest.json').exists():include(batch/'catalog/manifest.json')
    def verify(record,op):
        if not record:return dict(status='not_run')
        result=read(Path(record['result']))
        source=Path(record['source']);sha=hashlib.sha256(source.read_bytes()).hexdigest()
        if sha!=record['source_sha256']:raise ValueError('记录与源码不同 '+op)
        request=Path(record['result'].replace('.result.json','.request.json'))
        payload=read(request)
        if payload['implementation']['sources'][0]['content']!=source.read_text(encoding='utf8'):
            raise ValueError('请求与源码不同 '+op)
        include(source,sha);include(request);include(record['result'])
        initial_request=batch/'results'/op/'current-recheck/formal/HG-KDA-current-1.request.json'
        baseline_request=read(initial_request)
        for field in ['api_version','binding','settings']:
            if payload.get(field)!=baseline_request.get(field):
                raise ValueError('请求合同或计时参数不同 '+op+' '+field)
        timing=[r for r in result.get('per_workload',[]) if r.get('phase')=='timing']
        if result.get('status')=='PASSED':
            expected=[json.loads(line)['name'] for line in
                      (batch/contracts[op]['catalog_leaf']/'timing.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
            if sorted(r['uuid'] for r in timing)!=sorted(expected):
                raise ValueError('性能输入集合不完整 '+op)
            recomputed=math.exp(sum(math.log(r['reference_latency_ms']/r['latency_ms']) for r in timing)/len(timing))
            if not math.isclose(recomputed,result['geo_mean'],rel_tol=1e-9):
                raise ValueError('逐case加速比与汇总不同 '+op)
            if any(r.get('status')!='PASSED' for r in result.get('per_workload',[])):
                raise ValueError('通过状态与逐case结果矛盾 '+op)
        eligible=(result.get('status')=='PASSED' and result.get('num_passed')==result.get('num_workloads')
                  and not result.get('is_hack') and isinstance(result.get('geo_mean'),(int,float))
                  and math.isfinite(result['geo_mean']) and result['geo_mean']>0)
        supplement=None
        if record.get('extra_checks'):
            gates=read(Path(record['extra_checks']));include(record['extra_checks'])
            supplement=dict(status=gates.get('status'),passed=gates.get('passed'),total=gates.get('total'))
            eligible=eligible and gates.get('source_sha256')==sha and gates.get('operator')==op and gates.get('status')=='passed' and gates.get('passed')==gates.get('total') and bool(gates.get('total'))
            checker=gates.get('checker_source') or {}
            if checker.get('path'):include(checker['path'],checker.get('sha256'))
        else:eligible=False
        if record.get('native_manifest'):
            manifest=read(Path(record['native_manifest']));include(record['native_manifest'])
            for path,digest in manifest['files'].items():include(path,digest)
        if record.get('cross_safety'):
            safety=read(Path(record['cross_safety']));include(record['cross_safety'])
            eligible=eligible and safety.get('status')=='passed' and safety.get('source_sha256')==sha and safety.get('passed')==safety.get('total')
            include(safety['checker'],safety['checker_sha256'])
        if sha in denied:eligible=False
        return dict(status='qualified' if eligible else 'source_review_rejected' if sha in denied else 'not_qualified',score=result.get('geo_mean'),
                    complete=[result.get('num_passed'),result.get('num_workloads')],supplement=supplement,
                    source=source.relative_to(PROJECT).as_posix(),source_sha256=sha,
                    implementation_language=payload['implementation']['language'],backend=record.get('backend',record.get('reference_kind','Triton')),
                    result=str(record['result']),source_record=record)
    for record in plan['records']:
        op=record['operator'];contracts[op]=record;row=state.get('operators',{}).get(op,{})
        for name,sha in record['contract_sha256'].items():include(batch/record['catalog_leaf']/name,sha)
        initial=state.get('jobs',{}).get('verify-current:'+op,{}).get('result') or {}
        initial=verify(initial,op)
        best=verify(row.get('best'),op)
        confirmation=verify(row.get('final_confirmation'),op)
        if confirmation['status']!='not_run':
            if confirmation['source_sha256']!=best['source_sha256']:
                raise ValueError('最终确认与最终源码不同 '+op)
            best['confirmation']=confirmation
            if best['status']=='qualified' and confirmation['status']=='qualified':
                best['display_score']=math.sqrt(best['score']*confirmation['score'])
                best['stable_above_threshold']=min(best['score'],confirmation['score'])>.8
        reference=verify(state.get('references',{}).get(op),op)
        native=verify(state.get('native_candidates',{}).get(op),op)
        candidates=[]
        for key,job in state.get('jobs',{}).items():
            if not key.startswith('queued-proposal:'+op+':') or not job.get('result'):continue
            candidate=verify(job['result'],op)
            candidates.append(dict(job=key,**candidate))
        independent=[initial,best,confirmation]+candidates
        for key,job in state.get('jobs',{}).items():
            if key.startswith('search:'+op+':') and (job.get('result') or {}).get('result'):
                independent.append(verify(job['result'],op))
        from performance_summary import aggregate_independent
        best.update(aggregate_independent(independent,best.get('source_sha256')))
        best['independent_records']=[{name:entry[name] for name in
            ['status','score','source_sha256','result','source_record']} for key,entry in {
            entry['result']:entry for entry in independent
            if entry.get('status')=='qualified' and entry.get('source_sha256')==best.get('source_sha256')
        }.items()]
        rows.append(dict(operator=op,initial=initial,best=best,reference=reference,native_v1=native,
                         candidates=candidates,status=row.get('status')))
    proof=batch/'source-review/phase-index-proof.json'
    if proof.exists():
        value=read(proof);assert value['status']=='passed' and value['lanes_checked']==4096 and not value['failures']
        include(value['source'],value['source_sha256']);include(proof)
    for path in (batch/'source-review').glob('*/cross-safety-negative-control.json'):
        value=read(path);include(value['source'],value['source_sha256']);include(path)
        include(value['checker'],value['checker_sha256'])
    return dict(schema='hgkda-threeway-delivery-v1',status=state.get('status'),current_job=state.get('current_job'),
                time_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                rows=rows,files=files,external_source_archive=external,failed_jobs={k:v.get('error') for k,v in state.get('jobs',{}).items() if v.get('status')=='failed'},
                verification_operators=plan['verification_operators'],priority_operators=plan['priority_operators'])


def markdown(index):
    score=lambda r:'%.3f×'%r.get('display_score',r['score']) if isinstance(r.get('score'),(int,float)) else '—'
    counts=lambda r:'/'.join(str(x) for x in r.get('complete',[])) or '—'
    supplements=lambda r:('/'.join(str(r['supplement'][x]) for x in ['passed','total'])) if r.get('supplement') else '—'
    verified=[r for r in index['rows'] if r['operator'] in index['verification_operators']]
    good=sum(r['initial']['status']=='qualified' and r['initial'].get('score',0)>.8 for r in verified)
    lines=['# 本轮三路对照与算子优化','','截至北京时间'+index['time_beijing']+'。队列状态：'+str(index['status'])+'。',
        '',f'13项指定源码复验中，当前{good}项完整正确、补测通过且超过0.8×。正式计时沿用原KGS完整调用；当前队列状态不等于Goal完成。',
        '', '## 重点优化结果','', '|算子|本批起点|当前最佳|实现|完整评测|补测|状态|','|---|---:|---:|---|---|---|---|']
    by_op={row['operator']:row for row in index['rows']}
    for op in index['priority_operators']:
        row=by_op[op];best=row['best']
        lines.append('|'+op+'|'+score(row['initial'])+'|'+score(best)+'|'+best.get('backend','—')+'|'+counts(best)+'|'+supplements(best)+'|'+str(row['status'])+'|')
    lines+=['','## 历史成果同环境复验','','|算子|本批复验|完整评测|补测|结论|','|---|---:|---|---|---|']
    for row in verified:
        initial=row['initial'];passed=initial['status']=='qualified'
        status='达标' if passed and initial.get('score',0)>.8 else '正确，性能未达标' if passed else '待核验'
        lines.append('|'+row['operator']+'|'+score(initial)+'|'+counts(initial)+'|'+supplements(initial)+'|'+status+'|')
    lines+=['','## CANN原生库对照','','所有数值相对同环境PyTorch；PyTorch为1×。参考类型逐行列明。C++包装直接调用公开aclnn API，张量分配和任务队列适配使用torch_npu，不调用aten目标算子；Python ctypes包装另含逐调用转换与事件管理成本。完整调用均包含实际包装成本。较早版本保留在reference_history，不能当厂商最佳实现或物理上限。',
            '', '|算子|本批HG-KDA Triton起点|PyTorch|CANN参考|参考类型|CANN正确性|CANN补测|','|---|---:|---:|---:|---|---|---|']
    for row in index['rows']:
        if row['reference']['status']=='not_run':continue
        reference=row['reference']
        lines.append('|'+row['operator']+'|'+score(row['initial'])+'|1.000×|'+score(reference)+'|'+reference.get('backend','—')+'|'+counts(reference)+'|'+supplements(reference)+'|')
    lines+=['','## 新候选记录','','|算子|候选|成绩|完整评测|补测|资格|','|---|---|---:|---|---|---|']
    for row in index['rows']:
        for candidate in row['candidates']:
            lines.append('|'+row['operator']+'|'+candidate['job'].split(':',2)[-1]+'|'+score(candidate)+'|'+counts(candidate)+'|'+supplements(candidate)+'|'+candidate['status']+'|')
    lines+=['','三路参考先覆盖log10_、hardswish_out、native_group_norm。其余指定源码已安排原KGS复验；尚未接入独立原生库参考的项目不宣称已完成三路比较。',
            '', '## 方法与复现','',
            '- 使用相同原合同、源码SHA、输入规格、随机种子和KGS参数；参考与候选分开记账。最终确认对应同源码、同条件两份完整有效结果时，简表只显示几何平均，原始结果均保留；稳定过线要求两份均超过0.8×。',
            '- 自写Ascend C候选明确登记混合实现范围及C++源码、二进制SHA；不称为厂商最佳实现。',
            '- 设备路径在paths/；同进程恢复输入的log10补测在within-process/。独立profiling、宿主提交、含同步的单次完成耗时与正式KGS成绩分别解释。',
            '- 宿主tmux及物理卡7项目锁串行执行，不修改共享镜像和服务。原始失败、正式请求、逐case结果与补测保留。',
            '- 复现入口见../../../../tools/profiling/hgkda/threeway/README.md。',
            '', '## 待处理记录','']
    lines += ['- '+key+'：'+str(error) for key,error in index['failed_jobs'].items()] or ['当前无步骤异常。']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',default='threeway-20261030')
    args=parser.parse_args();batch=PROJECT/'operators/campaigns'/args.batch
    index=build(batch)
    (batch/'verified-progress.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (batch/'已核验进展.md').write_text(markdown(index),encoding='utf8')
    print(json.dumps(dict(status='verified',files=len(index['files']),rows=len(index['rows']),queue=index['status']),ensure_ascii=False))
