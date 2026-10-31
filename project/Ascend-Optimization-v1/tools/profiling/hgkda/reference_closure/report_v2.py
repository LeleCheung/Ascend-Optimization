"""核验源码、合同、请求和补测；原合同与各版CANN/NPU参考分列。"""
import hashlib
import io
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3];BATCH=PROJECT/'operators/campaigns/reference-closure-20261031'
sys.path.insert(0,str(HERE.parent/'next_phase'))
from report import verify
from validation_status import supplement_status
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    state=read(BATCH/'next-state.json');initial=read(BATCH/'starting-acceptance.json');evidence={};issues=[];rows=[]
    def track(p,expected=None):
        p=Path(p);digest=sha(p)
        if expected:assert digest==expected,'证据身份改变 '+str(p)
        evidence[str(p)]=digest
    track(initial['previous_delivery'],initial['sha256'])
    records={r['operator']:r for r in read(BATCH/'plan.json')['records']}
    for r in records.values():
        for n,d in r['contract_sha256'].items():track(BATCH/r['catalog_leaf']/n,d)
    def formal(key,op,leaf):
        job=state['jobs'].get(key,{});rec=job.get('result') or {}
        if not isinstance(rec,dict) or not rec.get('result'):return dict(status=job.get('status','pending'),error=job.get('error'))
        checked=verify(rec)
        if checked['status']!='verified':issues.append(checked);return checked
        source=Path(rec['source']);result=Path(rec['result']);request=result.with_name(result.name.replace('.result.json','.request.json'))
        track(source,rec['source_sha256']);track(result);track(request)
        payload=read(request);raw=read(result)
        assert not raw.get('is_hack') and payload['binding']['definition']==op
        bundle=result.parent/'bundle-response.json'
        if bundle.exists():assert read(bundle)['bundle_id']==payload['binding']['bundle_id'];track(bundle)
        from kernelgen_client.operator_bundles import pack_operator_bundle
        archive=io.BytesIO();bundle_sha,_=pack_operator_bundle(leaf,archive)
        assert payload['binding']['bundle_id']=='sha256:'+bundle_sha,'请求未绑定当前冻结合同'
        for n in ['definition.json','correctness.jsonl','timing.jsonl']:
            assert sha(leaf/n)==records[op]['contract_sha256'][n],'补充对照改变原输入或容差'
        for f in leaf.iterdir():
            if f.is_file():track(f)
        checked.update(backend=rec.get('backend'),reference_leaf=str(leaf),reference_oracle_sha256=sha(leaf/'oracle.py'))
        return checked
    def extra(p,op,source,leaf):
        if not p.exists():return dict(status='not_run')
        value=read(p);track(p);assert value['operator']==op
        checker=value.get('checker_source')
        if checker:track(checker['path'],checker['sha256'])
        command=value.get('reproduce_command',[])
        if '--leaf' in command:assert sha(Path(command[command.index('--leaf')+1])/'oracle.py')==sha(leaf/'oracle.py')
        return dict(status=supplement_status(op,value,sha(source)),passed=value.get('passed'),total=value.get('total'),evidence=str(p))
    for index in [BATCH/'reference-index.json']+sorted(BATCH.glob('reference-index-v*.json')):
        if not index.exists():continue
        track(index)
        for op,ref in read(index).items():
            version='' if ref.get('revision','v1')=='v1' else '-'+ref['revision'];source=Path(ref['source']);leaf=BATCH/records[op]['catalog_leaf']
            track(source,ref['source_sha256']);track(ref['binary'],ref['binary_sha256']);build=read(ref['build']);track(ref['build']);track(build['source'],build['source_sha256'])
            qpath=source.parent/'qualification.json';q=read(qpath) if qpath.exists() else {}
            if q:
                track(qpath);assert q['source_sha256']==sha(source) and q['contract_sha256']==records[op]['contract_sha256']
            qjob=state['jobs'].get('qualify'+version+':'+op,{})
            path_result=state['jobs'].get('cann-path'+version+':'+op,{}).get('result') or {}
            if path_result.get('trace'):track(path_result['trace'],path_result.get('trace_sha256'))
            rows.append(dict(operator=op,revision=ref.get('revision','v1'),kind=ref['kind'],scope=ref['scope'],
                qualification=dict(status=q.get('status',qjob.get('status','not_run')),passed=q.get('passed'),total=q.get('total'),eligible_full_contract=q.get('eligible_full_contract',False),error=qjob.get('error')),
                cann_vs_original=formal('cann-formal'+version+':'+op,op,leaf),cann_supplementary=extra(source.parent/'extra-checks.json',op,source,leaf),path=path_result,
                source=str(source),source_sha256=sha(source),binary_sha256=ref['binary_sha256']))
    comparisons=[]
    for op in ['fake_quantize_per_tensor_affine_cachemask_out','batch_norm_with_update_functional','fractional_max_pool3d','_conj_physical']:
        cann=op in ['fake_quantize_per_tensor_affine_cachemask_out','batch_norm_with_update_functional'];kind='CANN-direct' if cann else 'NPU-decomposition'
        key=('candidate-vs-cann:' if cann else 'supplemental-full:')+op
        if not cann and state['jobs'].get('supplemental-full-v2:'+op,{}).get('result'):
            key='supplemental-full-v2:'+op;kind='NPU-decomposition-v2'
        job=state['jobs'].get(key,{});rec=job.get('result') or {}
        if not isinstance(rec,dict) or not rec.get('result'):
            comparisons.append(dict(operator=op,reference=kind,status=job.get('status','pending'),error=job.get('error')));continue
        result=Path(rec['result']);label=result.name.split('-1.result.json')[0];leaf=BATCH/'catalog-snapshots'/('formal-'+label+'-'+op)/'ops'/op
        source=Path(rec['source']);checked=formal(key,op,leaf);gates=extra(BATCH/'comparisons'/op/kind/'extra-checks.json',op,source,leaf)
        ok=checked.get('correctness_passed') and gates['status']=='passed' and checked.get('score',0)>.8
        comparisons.append(dict(operator=op,reference=kind,formal=checked,supplementary=gates,threshold_passed=bool(ok),
            attribution='既有候选复测；本轮未新生成该实现',scope='独立命名NPU参考，不替代原PyTorch默认合同'))
    new_regular=[];searches={}
    for op in ['linalg_inv_ex','cumulative_trapezoid']:
        best=state.get('operators',{}).get(op,{}).get('best');checked=verify(best);searches[op]=checked
        if checked.get('status')=='verified' and checked.get('correctness_passed') and checked.get('extra_status')=='passed' and checked.get('score',0)>.8:
            if op=='cumulative_trapezoid':checked['reference_status']='默认参考稳定性仍需单列核验'
            else:new_regular.append(op)
    unknown=[]
    for request in BATCH.rglob('*.request.json'):
        payload=read(request)
        if payload.get('api_version')!='v6.2' or not payload.get('implementation'):continue
        result=request.with_name(request.name.replace('.request.json','.result.json'));track(request)
        if result.exists():track(result)
        else:unknown.append(str(request))
    accepted=list(dict.fromkeys(initial['accepted']+new_regular))
    data=dict(schema='hgkda-reference-closure-delivery-v2',queue_status=state.get('status'),current_job=state.get('current_job'),initial_regular=24,regular_accepted=len(accepted),total=30,
        new_regular=new_regular,accepted=accepted,references=rows,comparisons=comparisons,searches=searches,unresolved_requests=unknown,audit_issues=issues,evidence_sha256=evidence,
        rule='CANN/自建NPU参考不替代原PyTorch合同；资格失败、补测失败及参考冲突不能计为常规达标',attribution='Codex封装CANN参考和控制器；KG/CC/DeepSeek搜索以实际记录为准')
    (BATCH/'delivery.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
    def score(v):return '%.3f×'%v['score'] if isinstance(v.get('score'),(int,float)) else v.get('raw_status',v.get('status','未完成'))
    def count(v):return str(v.get('passed','—'))+'/'+str(v.get('total','—'))
    lines=['# 夜间六项与CANN参考交付','','原合同常规达标：'+str(len(accepted))+'/30；当前队列：'+str(state.get('status'))+'。',
        'CANN单算子、FP32兼容组合和NPU分解参考分列；换参考不自动增加原合同达标数。','',
        '|算子|CANN版本|资格|相对原PyTorch|额外检查|','|---|---|---|---|---|']
    for r in rows:lines.append('|'+r['operator']+'|'+r['revision']+' / '+r['kind']+'|'+r['qualification']['status']+' '+count(r['qualification'])+'|'+score(r['cann_vs_original'])+'|'+r['cann_supplementary']['status']+'|')
    lines+=['','## 候选与NPU参考','','|算子|参考|加速比|完整评测|额外检查|','|---|---|---|---|---|']
    for r in comparisons:
        f=r.get('formal',{});e=r.get('supplementary',{})
        lines.append('|'+r['operator']+'|'+r['reference']+'|'+score(f)+'|'+count(f)+'|'+e.get('status','未完成')+' '+count(e)+'|')
    lines+=['','## 原合同继续攻关','']
    for op,v in searches.items():lines.append('- '+op+'：'+score(v)+'；'+str(v.get('reference_status','详见搜索、原始结果和补测'))+'。')
    lines+=['','CANN包装由Codex实现，厂商实现只作参考；生成优化以KG/CC实际记录为准。',
        'expm1_原合同通过但极小数值补测失败，不能称扩展精度条件下已合格。所有失败版本保留。',
        '核验项：'+str(len(evidence))+'；未结束请求：'+str(len(unknown))+'；核验问题：'+str(len(issues))+'。']
    (BATCH/'交付报告.md').write_text(chr(10).join(lines)+chr(10),encoding='utf8')
    print(json.dumps(dict(regular_accepted=len(accepted),reference_rows=len(rows),comparisons=len(comparisons),evidence=len(evidence),unknown_requests=len(unknown),issues=len(issues))))
if __name__=='__main__':main()
