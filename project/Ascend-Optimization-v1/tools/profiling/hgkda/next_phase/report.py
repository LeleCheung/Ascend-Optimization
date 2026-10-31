"""按原始结果生成25项可汇报状态；不混用历史设备成绩与本轮walltime。"""
import argparse
import hashlib
import json
from pathlib import Path
from validation_status import supplement_status


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def reference_stability(batch,operator):
    if operator!='cumulative_trapezoid':return dict(status='not_applicable')
    root=batch/'analysis/cumulative_trapezoid/reference-stability-replay-v1'
    if not (root/'default.json').exists():return dict(status='not_run')
    try:
        default=read(root/'default.json');deterministic=read(root/'deterministic.json');meta=read(root/'provenance.json')
        binary=root/'input-fp16-256x8.bin';raw=binary.read_bytes()
        sha=hashlib.sha256(raw).hexdigest()
        assert default['input_sha256']==deterministic['input_sha256']==meta['input_sha256']==sha,'稳定性输入身份不符'
        assert default['shape']==[256,8] and default['dtype']=='float16' and len(raw)==4096,'冻结反例规格不符'
        script=root/'reproducer.py'
        if not script.exists():script=batch.parents[2]/'tools/profiling/hgkda/reproduce-cumsum-stability.py'
        assert hashlib.sha256(script.read_bytes()).hexdigest()==meta['script_sha256'],'稳定性脚本身份不符'
        assert default['deterministic'] is False and deterministic['deterministic'] is True,'参考执行条件未区分'
        for field in ['shape','dtype','dim','dx','repeats','torch_version','torch_npu_version','torch_git','physical_device']:
            assert default[field]==deterministic[field],'默认/确定性规格不同：'+field
        if default['empty_tolerance_intersections']:
            values=default['values'];assert len(values)==default['repeats']>=2
            lower=max(x-(.015+.015*abs(x)) for x in values)
            upper=min(x+(.015+.015*abs(x)) for x in values)
            assert lower>upper,'给出的数值未构成无公共容差区间反例'
            return dict(status='default_reference_inconsistent',empty_intersections=default['empty_tolerance_intersections'],
                counterexample_position=default['position'],common_lower=lower,common_upper=upper,
                deterministic_consistent_in_tolerance=deterministic['empty_tolerance_intersections']==0 and
                    not any(deterministic['over_tolerance_vs_first']),
                input_sha256=sha,evidence=str(root/'default.json'),
                interpretation='这些默认参考之间不存在共同满足原0.015容差的固定输出；确定性条件单列')
        return dict(status='no_counterexample_in_this_run',input_sha256=sha)
    except (OSError,ValueError,KeyError,AssertionError) as error:
        return dict(status='evidence_requires_review',error=str(error))


def verify(record):
    if not record:return dict(status='pending')
    source=Path(record['source']);result=Path(record['result'])
    request=result.with_name(result.name.replace('.result.json','.request.json'))
    try:
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest==record['source_sha256'],'源码SHA改变'
        raw=read(result);payload=read(request)
        sources=payload['implementation']['sources']
        assert len(sources)==1 and sources[0]['content']==source.read_text(encoding='utf8'),'请求未绑定此源码'
        passed=raw.get('status')=='PASSED' and raw.get('num_passed')==raw.get('num_workloads') and raw.get('num_workloads',0)>0
        extra=read(Path(record['extra_checks'])) if record.get('extra_checks') else None
        if extra is not None:
            assert extra['source_sha256']==digest,'补充检查源码不同'
        return dict(status='verified',correctness_passed=passed,score=raw.get('geo_mean') if passed else None,
            passed=raw.get('num_passed'),total=raw.get('num_workloads'),source_sha256=digest,source=str(source),
            result=str(result),result_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),request=str(request),
            extra_status=supplement_status(extra.get('operator'),extra,digest) if extra else 'not_run',raw_status=raw.get('status'))
    except (OSError,ValueError,KeyError,AssertionError) as error:
        return dict(status='evidence_requires_review',error=str(error),record=record)


def generate(batch,quiet=False):
    state=read(batch/'next-state.json');targets=read(batch/'targets.json')['targets']
    rows=[]
    for target in targets:
        op=target['operator'];current=state.get('operators',{}).get(op,{})
        base=verify(current.get('baseline'));start=verify(current.get('starting_point'));best=verify(current.get('best'))
        if base.get('source_sha256') and base['source_sha256']!=target['source_sha256']:
            from baseline import resolve
            try:
                spec=resolve(batch,op,dict(seed='seeds/'+op+'.py',source_sha256=target['source_sha256']))
                assert spec.get('adapter') and spec['source_sha256']==base['source_sha256'],'FlagGems列不是原版或核验入口适配'
                base['entry_adapter']=spec['adapter']
            except (OSError,ValueError,KeyError,AssertionError) as error:
                base.update(status='evidence_requires_review',error=str(error))
        candidate=best.get('source_sha256') and best.get('source_sha256')!=base.get('source_sha256')
        reused=bool(candidate and best.get('source_sha256')==start.get('source_sha256'))
        deliverable=bool(candidate and best.get('correctness_passed') and best.get('extra_status')=='passed' and best.get('score',0)>.8)
        if deliverable:
            status='本批通过，代表参考路径已审计' if current.get('reference_path_status')=='npu_paths_observed' else '本批数值和性能通过，参考路径需核验'
        elif candidate and best.get('correctness_passed'):
            status='补充检查失败' if best.get('extra_status')=='failed' else '性能未达标' if best.get('score',0)<=.8 else '待补充检查'
        else:status=current.get('status',target['status'])
        if reused:status='保留历史实现；'+status
        if status=='contract_prepared_pending_validation':status='合同已准备，待运行'
        stability=reference_stability(batch,op)
        stability_qualified=stability['status'] not in {'default_reference_inconsistent','evidence_requires_review'}
        if stability['status']=='default_reference_inconsistent':
            status='默认参考重复不自洽；确定性条件需单列'
        elif not stability_qualified:status='参考稳定性证据需复核'
        rows.append(dict(order=target['order'],operator=op,flaggems=base,historical_start=start,hgkda=best if candidate else dict(status='not_new'),
            threshold_passed=bool(deliverable and current.get('reference_path_status')=='npu_paths_observed'
                                  and stability_qualified),reference_stability=stability,
            numerical_threshold_passed=deliverable,comparison_conditions_require_review=bool(deliverable and
                (current.get('reference_path_status')!='npu_paths_observed' or not stability_qualified)),
            reused_historical_source=reused,status=status,reference_path_status=current.get('reference_path_status','not_run')))
    result=dict(schema='hgkda-next25-report-v1',timing_scope='本轮KGS19659 walltime完整调用；不与历史device_kernel加速比混用',
        queue_status=state['status'],current_job=state.get('current_job'),threshold=.8,operators=rows,
        software_lock=read(batch/'software-lock.json'),source_state_sha256=hashlib.sha256((batch/'next-state.json').read_bytes()).hexdigest())
    def score(row):
        if row.get('score') is not None:return f"{row['score']:.3f}×"
        if row.get('total'):return str(row.get('raw_status'))+' '+str(row['passed'])+'/'+str(row['total'])
        return '—'
    lines=['# 新25算子夜间进展','',result['timing_scope']+'。原版列绑定固定FlagGems适配源码，历史候选单列；本批代表合同不冒称上游全部pytest。','',
        '|算子|FlagGems实测|历史优化起点|HG-KDA当前|评测|补充检查|状态|','|---|---:|---:|---:|---|---|---|']
    for row in rows:
        best=row['hgkda'];tests=str(best['passed'])+'/'+str(best['total']) if best.get('total') else '—'
        lines.append('|'+row['operator']+'|'+score(row['flaggems'])+'|'+score(row['historical_start'])+'|'+score(best)+'|'+tests+'|'+best.get('extra_status','—')+'|'+row['status']+'|')
    (batch/'verified-progress.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (batch/'已核验进展.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    if not quiet:print(json.dumps(dict(operators=len(rows),threshold_passed=sum(r['threshold_passed'] for r in rows),queue_status=state['status'],current_job=state.get('current_job')),ensure_ascii=False))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True);generate(p.parse_args().batch)
