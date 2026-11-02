"""按后端资格、原语义与合格CANN参考重建10＋20表，不沿用混合后端计数。"""
import math
from pathlib import Path
from common import BATCH,HERE,PREVIOUS,read,save,sha
from gate import admit
from verify_compiled import validate_manifest

def main():
    state=read(BATCH/'state.json');index=read(BATCH/'candidate-index.json');old=read(PREVIOUS/'delivery.json')
    rows=[];evidence={}
    def track(p):p=Path(p);evidence[str(p)]=sha(p);return p
    for previous in old['rows']:
        op=previous['operator'];v=state['operators'].get(op,{})
        source=Path(index[op]['source']);audit=admit(source,source.parent/'review.json')
        track(source);track(source.parent/'review.json')
        runtime=v.get('runtime') or {};extra=v.get('supplementary') or {};pt=v.get('pytorch') or {};ca=v.get('cann') or {};ce=v.get('cann_supplementary') or {}
        for k in ['runtime','supplementary','pytorch','cann','cann_supplementary']:
            item=v.get(k) or {};path=item.get('path')
            if path:
                raw=read(track(path))
                if k in {'pytorch','cann'}:
                    item['is_hack']=raw.get('is_hack')
                    request=read(track(Path(path).with_name(Path(path).name.replace('.result.json','.request.json'))))
                    assert request['implementation']['sources'][0]['content']==source.read_text(encoding='utf8')
                    assert request['settings']==dict(warmup_ms=1000,benchmark_ms=100,num_trials=3,timeout_seconds=1800)
                    details=[x for x in raw.get('per_workload',[]) if x.get('phase')=='timing' and x.get('latency_ms') and x.get('reference_latency_ms')]
                    if raw.get('status')=='PASSED':
                        assert raw['num_passed']==raw['num_workloads']>0
                        assert len(raw['per_workload'])==raw['num_workloads'] and all(v.get('status')=='PASSED' for v in raw['per_workload'])
                        score=math.exp(sum(math.log(x['reference_latency_ms']/x['latency_ms']) for x in details)/len(details))
                        assert math.isclose(score,item['score'],rel_tol=1e-9)
                else:assert raw['source_sha256']==sha(source)
        numerical=pt.get('status')=='PASSED' and not pt.get('is_hack') and extra.get('status')=='passed'
        backend=runtime.get('status')=='passed'
        compiled=v.get('compiled_path') or {}
        compiled_ok=compiled.get('status')=='verified' and compiled.get('source_sha256')==sha(source)
        if compiled.get('path'):
            proof=read(track(compiled['path']))
            assert proof['source_sha256']==sha(source)
            assert proof==validate_manifest(Path(proof['manifest']),source),'编译路径证明与实际产物不符'
            for artifact,digest in proof.get('artifacts_sha256',{}).items():
                assert sha(track(artifact))==digest
        accepted=bool(numerical and backend and ca.get('status')=='PASSED' and not ca.get('is_hack') and ce.get('status')=='passed' and ca.get('score',0)>.8 and compiled_ok)
        status='严格Triton达标' if accepted else ('尚未完成' if not v else '待复核或未达标')
        if v.get('status')=='运行时后端或执行失败，待修复':status=v['status']
        elif v.get('status')=='本轮已复核' and not accepted:
            if not numerical:status='原语义或补测未通过'
            elif not previous.get('cann_reference_qualified'):status='原语义通过，CANN参考不合格或缺失，单列'
            elif ca.get('status')!='PASSED' or ce.get('status')!='passed':status='CANN语义对照未通过'
            elif ca.get('score',0)<=.8:status='严格Triton正确，性能未过0.8×'
            elif not compiled_ok:status='数值与性能通过，编译路径待核验'
        path=v.get('device_path') or {}
        if path.get('trace'):track(path['trace']);track(path['summary'])
        rows.append(dict(order=previous['order'],group='1015' if previous['order']<=10 else '1029',operator=op,
            strict_backend=backend,accepted=accepted,status=status,source=str(source),source_sha256=sha(source),
            pytorch=pt,cann=ca,supplementary=extra,cann_supplementary=ce,reference_kind=previous.get('reference_kind'),
            reference_status='合格' if previous.get('cann_reference_qualified') else previous['cann_status'],
            historical_mixed_status=previous['cann_status'],review=index[op],runtime=runtime,device_path=path,compiled_path=compiled))
    groups={name:dict(total=sum(r['group']==name for r in rows),accepted=sum(r['group']==name and r['accepted'] for r in rows)) for name in ['1015','1029']}
    result=dict(schema='hgkda-strict-triton-delivery-v1',queue_status=state['status'],current_job=state.get('current_job'),
        accepted=sum(r['accepted'] for r in rows),total=30,groups=groups,rows=rows,evidence_sha256=evidence,
        policy='候选全部设备计算与搬运严格Triton；CANN/Ascend C仅参考；原PyTorch语义和补测保留；>0.8验收，>1快于参考')
    save(BATCH/'delivery.json',result)
    implementations=[]
    references=[]
    for r in rows:
        if r['strict_backend'] and r['pytorch'].get('status')=='PASSED' and not r['pytorch'].get('is_hack') and r['supplementary'].get('status')=='passed' and r['compiled_path'].get('status')=='verified':
            implementations.append(dict(operator=r['operator'],source=r['source'],source_sha256=r['source_sha256'],
                review=str(Path(r['source']).parent/'review.json'),runtime=r['runtime'],compiled_path=r['compiled_path'],
                formal=r['pytorch'],supplementary=r['supplementary'],cann=r['cann'],cann_supplementary=r['cann_supplementary'],
                target_accepted=r['accepted'],usage='仅作已核验Triton实现上下文；新算子独立评测，引用前重新核验源码和证据SHA'))
        historical=next(v for v in old['rows'] if v['operator']==r['operator'])
        if historical.get('cann_reference_qualified'):
            references.append(dict(operator=r['operator'],source=historical['reference_source'],
                source_sha256=historical['reference_source_sha256'],kind=r['reference_kind'],
                usage='仅用于参考侧成对对照和机制分析，不作为候选模板或参与候选择优'))
    save(BATCH/'strict-implementation-context.json',dict(schema='hgkda-strict-implementation-context-v1',
        delivery_sha256=sha(BATCH/'delivery.json'),implementations=implementations,references=references,
        policy='实现模板只收录逐路径和依赖审查、运行时检查、原KGS与补测以及实际Triton编译产物核验通过的源码；历史混合后端不进入实现模板'))
    lines=['# 严格Triton 10＋20复核','','当前严格Triton达标%d/30；队列%s。'%(result['accepted'],state['status']),
        '所有加速比为KGS完整调用；合格CANN单接口与兼容／组合分别标注。历史混合实现不计入。','']
    for group in ['1015','1029']:
        lines+=['## '+group+'（%d/%d）'%(groups[group]['accepted'],groups[group]['total']),'',
            '|算子|候选后端|参考类型|相对CANN|相对PyTorch|原合同|补测|状态|','|---|---|---|---:|---:|---|---|---|']
        for r in rows:
            if r['group']!=group:continue
            fmt=lambda x:'%.3f×'%x['score'] if x.get('status')=='PASSED' and x.get('score') is not None else '—'
            p=r['pytorch'];complete='%s/%s'%(p.get('passed','—'),p.get('total','—'))
            kind={'CANN_native_single_api':'CANN单接口',
                  'CANN_layout_compatibility':'CANN布局适配',
                  'CANN_dtype_compatibility':'CANN精度适配',
                  'CANN_native_uniform_with_generator_adapter':'CANN随机状态适配',
                  'CANN_zeros_scatter_composition':'CANN清零＋回填',
                  'CANN_Copy_Neg_storage_composition':'CANN复制＋取负',
                  'CANN_FP32_compatibility_composition':'CANN FP32兼容组合',
                  'CANN_FP32_state_compatibility_composition':'CANN FP32状态组合',
                  'CANN_Add_Muls_Cumsum_composition':'CANN累计积分组合',
                  }.get(r['reference_kind'],r['reference_kind'] or '无同语义参考')
            backend='Triton＋C++宿主' if r['review'].get('reviewed_host_bridges') else 'Triton'
            gates=lambda v:('%s/%s'%(v['passed'],v['total'])) if v.get('passed') is not None and v.get('total') else v.get('status','—')
            extra='原：'+gates(r['supplementary'])+'；CANN：'+gates(r['cann_supplementary'])
            source=Path(r['source']).relative_to(BATCH).as_posix()
            lines.append('|['+r['operator']+']('+source+')|'+backend+'|'+kind+'|'+fmt(r['cann'])+'|'+fmt(p)+'|'+complete+'|'+extra+'|'+r['status']+'|')
        lines.append('')
    lines+=['`fractional_max_pool3d`、`_conj_physical`的PyTorch参考含CPU回退；该列只保留原合同结果，不能解释为相对纯NPU实现的收益。',
        '算子名链接本轮冻结源码；来源、审查、归属与SHA见`candidate-index.json`及每项`review.json`。完整原始成绩和复用标记见`delivery.json`。',
        '所有门槛使用未舍入的原始加速比判断。参考资格、原正确性、补测、后端与编译路径任何一项未通过，均不计为本轮达标。','']
    (BATCH/'严格Triton30项总表.md').write_bytes((chr(10).join(lines)+chr(10)).encode('utf8'))
    print('STRICT_REPORT',result['accepted'],groups,flush=True)

if __name__=='__main__':main()
