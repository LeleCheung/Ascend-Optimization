"""将完整评测转为继续优化决策和代表case；不把端到端延迟直接归因硬件组件。"""
import math


def performance_feedback(result, timing_cases, *, threshold=.8, checks_passed=False):
    contracts={case['name']:case for case in timing_cases}
    rows=[row for row in result.get('per_workload',[]) if row.get('phase')=='timing']
    full=(result.get('status')=='PASSED' and result.get('num_passed')==result.get('num_workloads')
          and result.get('num_workloads',0)>0 and result.get('is_hack') is False)
    if not full:
        return dict(decision='repair_correctness_or_runtime',accepted=False,selected_cases=[],
                    formal_status=result.get('status'),reason='完整正式评测未通过，先修复执行或语义')
    if len(rows)!=len(contracts) or {row['uuid'] for row in rows}!=set(contracts):
        raise ValueError('性能结果与冻结case清单不一致')
    enriched=[]
    for row in rows:
        speedup=row.get('speedup')
        if not isinstance(speedup,(float,int)) or not math.isfinite(speedup) or speedup<=0:
            raise ValueError('性能比值必须为有限正数')
        spec=contracts[row['uuid']].get('inputs',{}).get('spec',{})
        shape=spec.get('shape',[])
        enriched.append(dict(case=row['uuid'],dtype=spec.get('dtype'),shape=shape,
            elements=math.prod(shape),speedup=speedup,log_loss=-math.log(speedup),
            candidate_us=row['latency_ms']*1000,reference_us=row['reference_latency_ms']*1000))
    geo=math.exp(sum(math.log(row['speedup']) for row in enriched)/len(enriched))
    if not math.isclose(geo,result['geo_mean'],rel_tol=1e-9):
        raise ValueError('完整性能无法由逐case重建')
    selected=[]
    for dtype in dict.fromkeys(row['dtype'] for row in enriched):
        group=[row for row in enriched if row['dtype']==dtype]
        for row in [max(group,key=lambda r:r['log_loss']),max(group,key=lambda r:r['elements'])]:
            if row not in selected:selected.append(row)
    accepted=geo>threshold and checks_passed
    decision=('target_met' if accepted else 'complete_supplementary' if geo>threshold else 'continue_optimization')
    return dict(decision=decision,accepted=accepted,geo_mean=geo,threshold=threshold,
                selected_cases=selected,ranked_cases=sorted(enriched,key=lambda r:r['log_loss'],reverse=True),
                interpretation='按几何平均损失选择case，并补充各类型最大输入；组件原因仍需独立取证')
