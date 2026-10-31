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
        timing=[r for r in raw.get('per_workload',[]) if r.get('phase')=='timing']
        checked['timing_summary']=dict(total=len(timing),below_threshold=sum(isinstance(r.get('speedup'),(int,float)) and r['speedup']<.8 for r in timing),
            worst=[{k:r.get(k) for k in ['uuid','speedup','latency_ms','reference_latency_ms']} for r in sorted(timing,key=lambda r:r.get('speedup') if isinstance(r.get('speedup'),(int,float)) else float('inf'))[:3]])
        return checked
    def extra(p,op,source,leaf):
        if not p.exists():return dict(status='not_run')
        value=read(p);track(p);assert value['operator']==op
        checker=value.get('checker_source')
        if checker:track(checker['path'],checker['sha256'])
        command=value.get('reproduce_command',[])
        if '--leaf' in command:assert sha(Path(command[command.index('--leaf')+1])/'oracle.py')==sha(leaf/'oracle.py')
        return dict(status=supplement_status(op,value,sha(source)),passed=value.get('passed'),total=value.get('total'),evidence=str(p))
    def dynamic(op,source,leaf):
        p=source.parent/'dynamic-semantics.json'
        if not p.exists():return dict(status='not_run')
        value=read(p);track(p)
        assert value['operator']==op and value['source_sha256']==sha(source)
        assert value['contract_sha256']=={n:sha(leaf/n) for n in ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']}
        track(value['checker_source']['path'],value['checker_source']['sha256'])
        assert value['total']==9 and len(value['cases'])==9
        assert value['passed']==sum(bool(r['passed']) for r in value['cases'])
        return dict(status='passed' if value['passed']==9 else 'failed',passed=value['passed'],total=9,evidence=str(p),
            failures=[r for r in value['cases'] if not r['passed']])
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
                source=str(source),source_sha256=sha(source),binary_sha256=ref['binary_sha256'],dynamic_semantics=dynamic(op,source,leaf)))
    comparisons=[]
    for op in ['fix','fake_quantize_per_tensor_affine_cachemask_out','batch_norm_with_update_functional','fractional_max_pool3d','_conj_physical']:
        cann=op in ['fix','fake_quantize_per_tensor_affine_cachemask_out','batch_norm_with_update_functional'];kind='CANN-direct' if cann else 'NPU-decomposition'
        key=('candidate-vs-cann:' if cann else 'supplemental-full:')+op
        if not cann and state['jobs'].get('supplemental-full-v2:'+op,{}).get('result'):
            key='supplemental-full-v2:'+op;kind='NPU-decomposition-v2'
        job=state['jobs'].get(key,{});rec=job.get('result') or {}
        if not isinstance(rec,dict) or not rec.get('result'):
            comparisons.append(dict(operator=op,reference=kind,status=job.get('status','pending'),error=job.get('error')));continue
        result=Path(rec['result']);label=result.name.split('-1.result.json')[0];leaf=BATCH/'catalog-snapshots'/('formal-'+label+'-'+op)/'ops'/op
        source=Path(rec['source']);checked=formal(key,op,leaf);gates=extra(BATCH/'comparisons'/op/kind/'extra-checks.json',op,source,leaf)
        reference_dynamic=dict(status='not_applicable')
        if op in ['fake_quantize_per_tensor_affine_cachemask_out','batch_norm_with_update_functional']:
            revision='v2' if op.startswith('fake_') else 'v3'
            reference=read(BATCH/('reference-index-'+revision+'.json'))[op]
            reference_dynamic=dynamic(op,Path(reference['source']),BATCH/records[op]['catalog_leaf'])
        ok=checked.get('correctness_passed') and gates['status']=='passed' and checked.get('score',0)>.8 and reference_dynamic['status'] in {'passed','not_applicable'}
        comparisons.append(dict(operator=op,reference=kind,formal=checked,supplementary=gates,threshold_passed=bool(ok),
            reference_dynamic_semantics=reference_dynamic,
            attribution='既有候选复测；本轮未新生成该实现',scope='独立命名NPU参考，不替代原PyTorch默认合同'))
    isolated_path=BATCH/'references/linalg_inv_ex/isolated-audit-v1/summary.json'
    inverse_isolated=read(isolated_path) if isolated_path.exists() else dict(status='pending')
    if isolated_path.exists():
        track(isolated_path)
        assert inverse_isolated['source_sha256']==sha(inverse_isolated['source']) and inverse_isolated['eligible_full_inv_ex_contract'] is False
        for f in isolated_path.parent.rglob('*'):
            if f.is_file():track(f)
    proposal=None;proposal_source=BATCH/'proposals/linalg_inv_ex/padded-block-v1/main.py'
    if proposal_source.exists():
        track(proposal_source);directory=proposal_source.parent;op='linalg_inv_ex'
        for path in directory.glob('*.json'):track(path)
        diagnostic=read(directory/'diagnostic.json') if (directory/'diagnostic.json').exists() else dict(status='pending')
        if diagnostic.get('source_sha256'):assert diagnostic['source_sha256']==sha(proposal_source)
        proposal=dict(attribution='Codex修改CC候选的一般性最小分块；非KG独立自动结果',diagnostic=diagnostic,
            formal=formal('inverse-padded-formal',op,BATCH/'catalog-snapshots/formal-HG-KDA-padded-block-linalg_inv_ex/ops/linalg_inv_ex'),
            supplementary=extra(directory/'extra-checks.json',op,proposal_source,BATCH/records[op]['catalog_leaf']))
    scan_proposal=None;scan_source=BATCH/'proposals/cumulative_trapezoid/layout-restore-v1/main.py'
    if scan_source.exists():
        track(scan_source);directory=scan_source.parent;op='cumulative_trapezoid'
        if (directory/'proposal.json').exists():track(directory/'proposal.json')
        scan_proposal=dict(attribution='CC生成第三轮候选＋Codex通用布局还原修复',
            formal=formal('scan-layout-repair-formal',op,BATCH/'catalog-snapshots/formal-HG-KDA-layout-restore-cumulative_trapezoid/ops/cumulative_trapezoid'),
            supplementary=extra(directory/'extra-checks.json',op,scan_source,BATCH/records[op]['catalog_leaf']),
            reference_status='原默认PyTorch合同；参考重复稳定性问题独立保留')
    scan_v2=None;scan_source_v2=BATCH/'proposals/cumulative_trapezoid/tiled-restore-v2/main.py'
    if scan_source_v2.exists():
        track(scan_source_v2);directory=scan_source_v2.parent;op='cumulative_trapezoid'
        if (directory/'proposal.json').exists():track(directory/'proposal.json')
        scan_v2=dict(attribution='Codex在CC候选上修复首轴分块还原；不是纯自动生成结果',
            formal=formal('scan-tiled-restore-v2-formal',op,BATCH/'catalog-snapshots/formal-HG-KDA-tiled-restore-v2-cumulative_trapezoid/ops/cumulative_trapezoid'),
            supplementary=extra(directory/'extra-checks.json',op,scan_source_v2,BATCH/records[op]['catalog_leaf']),
            reference_status='默认参考稳定性问题独立保留')
        conditional=directory/'deterministic-condition';conditions=conditional/'conditions.json';conditional_result=conditional/'deterministic-1.result.json'
        if conditions.exists() and conditional_result.exists():
            c=read(conditions);raw=read(conditional_result);request=conditional/'deterministic-1.request.json';payload=read(request)
            assert c['source_sha256']==sha(scan_source_v2) and c['deterministic'] is True and c['applies_to']=='reference_and_candidate'
            assert c['contract_sha256']==records[op]['contract_sha256'] and payload['implementation']['sources'][0]['content']==scan_source_v2.read_text()
            for path in [conditions,conditional_result,request,conditional/'preflight.json']:track(path)
            gates=extra(conditional/'extra-checks.json',op,scan_source_v2,BATCH/records[op]['catalog_leaf'])
            scan_v2['deterministic_condition']=dict(status=raw.get('status'),score=raw.get('geo_mean'),passed=raw.get('num_passed'),total=raw.get('num_workloads'),
                supplementary=gates,scope='官方确定性同时适用于参考与候选；不计入默认24项常规交付')
    scan_hybrid=None;hybrid_source=BATCH/'proposals/cumulative_trapezoid/hybrid-cann-v3/main.py'
    if hybrid_source.exists():
        op='cumulative_trapezoid';directory=hybrid_source.parent;track(hybrid_source);track(directory/'proposal.json')
        build=read(BATCH/'cann-cumsum-build.json');track(BATCH/'cann-cumsum-build.json');track(build['source'],build['source_sha256']);track(build['binary'],build['binary_sha256'])
        scan_hybrid=dict(attribution='CC末轴Triton扫描＋Codex非末轴增量及直接CANN扫描桥；混合实现',
            formal=formal('scan-hybrid-v3-formal',op,BATCH/'catalog-snapshots/formal-HG-KDA-hybrid-cann-v3-cumulative_trapezoid/ops/cumulative_trapezoid'),
            supplementary=extra(directory/'extra-checks.json',op,hybrid_source,BATCH/records[op]['catalog_leaf']))
        conditional=directory/'deterministic-condition';conditions=conditional/'conditions.json';conditional_result=conditional/'deterministic-1.result.json'
        if conditions.exists() and conditional_result.exists():
            c=read(conditions);raw=read(conditional_result);request=conditional/'deterministic-1.request.json';payload=read(request)
            assert c['source_sha256']==sha(hybrid_source) and c['deterministic'] is True and c['applies_to']=='reference_and_candidate'
            assert c['contract_sha256']==records[op]['contract_sha256'] and payload['implementation']['sources'][0]['content']==hybrid_source.read_text()
            for path in [conditions,conditional_result,request,conditional/'preflight.json']:track(path)
            gates=extra(conditional/'extra-checks.json',op,hybrid_source,BATCH/records[op]['catalog_leaf'])
            scan_hybrid['deterministic_condition']=dict(status=raw.get('status'),score=raw.get('geo_mean'),passed=raw.get('num_passed'),total=raw.get('num_workloads'),
                supplementary=gates,scope='双方官方确定性；厂商扫描混合实现；不计默认稳定达标')
    new_regular=[];searches={}
    for op in ['linalg_inv_ex','cumulative_trapezoid']:
        best=state.get('operators',{}).get(op,{}).get('best');checked=verify(best);searches[op]=checked
        manifest=BATCH/'results'/op/'search-1/native-evidence/archive-manifest.json'
        if manifest.exists():
            native=read(manifest);track(manifest);rounds=[]
            for item in native['files']:
                path=manifest.parent/item['archived_path'];track(path,item['sha256'])
                if '/round-' in item['source_path'] and item['source_path'].endswith('/result.json'):
                    raw=read(path);rounds.append(dict(round=item['source_path'],status=raw.get('status'),passed=raw.get('num_passed'),
                        total=raw.get('num_workloads'),score=raw.get('geo_mean') if raw.get('status')=='PASSED' else None,evidence=str(path)))
            checked['native_search']=dict(state=native.get('state'),rounds=rounds,manifest=str(manifest))
        job=state['jobs'].get('search:'+op+':1',{})
        if not best:checked['status']=job.get('result',{}).get('status',job.get('status','pending'))
        if checked.get('status')=='verified' and checked.get('correctness_passed') and checked.get('extra_status')=='passed' and checked.get('score',0)>.8:
            checked['original_numerical_contract_passed']=True
            if op=='cumulative_trapezoid':checked['reference_status']='默认参考稳定性仍需单列核验'
            else:checked['reference_status']='原PyTorch参考含CPU回退，数值通过与公平NPU性能分列'
    scan_paths=[]
    for directory in sorted((BATCH/'paths/cumulative_trapezoid').glob('hybrid-v3-*')):
        spec=read(directory/'spec.json');assert spec['source_sha256']==sha(hybrid_source) and spec['deterministic'] is True
        track(directory/'spec.json')
        for side in ['candidate','pytorch']:
            path=directory/(side+'-summary.json')
            if path.exists():
                summary=read(path);track(path);track(summary['trace'],summary['trace_sha256'])
                scan_paths.append(dict(case=spec['case']['name'],side=side,**{k:v for k,v in summary.items() if k!='device_tasks'}))
        manifest=directory/'candidate-ir/manifest.json'
        if manifest.exists():
            value=read(manifest);track(manifest);assert value['entries'] and value['spec']==spec
            for entry in value['entries']:
                for artifact in entry['artifacts']:track(artifact['path'],artifact['sha256'])
    unknown=[]
    for request in BATCH.rglob('*.request.json'):
        payload=read(request)
        if payload.get('api_version')!='v6.2' or not payload.get('implementation'):continue
        result=request.with_name(request.name.replace('.request.json','.result.json'));track(request)
        if result.exists():track(result)
        else:unknown.append(str(request))
    accepted=list(dict.fromkeys(initial['accepted']+new_regular))
    data=dict(schema='hgkda-reference-closure-delivery-v2',queue_status=state.get('status'),current_job=state.get('current_job'),initial_regular=24,regular_accepted=len(accepted),total=30,
        new_regular=new_regular,accepted=accepted,references=rows,comparisons=comparisons,searches=searches,inverse_isolated=inverse_isolated,inverse_padded_proposal=proposal,scan_layout_proposal=scan_proposal,scan_tiled_proposal=scan_v2,scan_hybrid_proposal=scan_hybrid,scan_execution_paths=scan_paths,unresolved_requests=unknown,audit_issues=issues,evidence_sha256=evidence,
        rule='CANN/自建NPU参考不替代原PyTorch合同；资格失败、补测失败及参考冲突不能计为常规达标',attribution='Codex封装CANN参考和控制器；KG/CC/DeepSeek搜索以实际记录为准')
    (BATCH/'delivery.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
    def score(v):return '%.3f×'%v['score'] if isinstance(v.get('score'),(int,float)) else v.get('raw_status',v.get('status','未完成'))
    def count(v):return str(v.get('passed','—'))+'/'+str(v.get('total','—'))
    lines=['# 夜间六项与CANN参考交付','','原合同常规达标：'+str(len(accepted))+'/30；当前队列：'+str(state.get('status'))+'。',
        'CANN单算子、FP32兼容组合和NPU分解参考分列；换参考不自动增加原合同达标数。','',
        '|算子|CANN版本|资格|相对原PyTorch|额外检查|动态参数检查|','|---|---|---|---|---|---|']
    for r in rows:lines.append('|'+r['operator']+'|'+r['revision']+' / '+r['kind']+'|'+r['qualification']['status']+' '+count(r['qualification'])+'|'+score(r['cann_vs_original'])+'|'+r['cann_supplementary']['status']+'|'+r['dynamic_semantics']['status']+' '+count(r['dynamic_semantics'])+'|')
    lines+=['','## 候选与NPU参考','','|算子|参考|加速比|完整评测|额外检查|','|---|---|---|---|---|']
    for r in comparisons:
        f=r.get('formal',{});e=r.get('supplementary',{})
        lines.append('|'+r['operator']+'|'+r['reference']+'|'+score(f)+'|'+count(f)+'|'+e.get('status','未完成')+' '+count(e)+'|')
    lines+=['','## 原合同继续攻关','']
    for op,v in searches.items():lines.append('- '+op+'：'+score(v)+'；'+str(v.get('reference_status','详见搜索、原始结果和补测'))+'。')
    if inverse_isolated.get('total'):lines.append('- CANN求逆逐case隔离：逆矩阵数值通过'+str(inverse_isolated['inverse_values_passed'])+'/'+str(inverse_isolated['total'])+'；aclnnInverse不提供info，仍不是完整inv_ex参考。')
    if proposal:lines.append('- 求逆最小分块候选：诊断'+count(proposal['diagnostic'])+'；正式'+score(proposal['formal'])+'；补测'+proposal['supplementary']['status']+'。')
    if scan_proposal:lines.append('- 累计积分布局修复候选：正式'+score(scan_proposal['formal'])+' '+count(scan_proposal['formal'])+'；补测'+scan_proposal['supplementary']['status']+' '+count(scan_proposal['supplementary'])+'；默认参考稳定性问题单列。')
    if scan_v2:lines.append('- 累计积分分块还原候选：正式'+score(scan_v2['formal'])+' '+count(scan_v2['formal'])+'；补测'+scan_v2['supplementary']['status']+' '+count(scan_v2['supplementary'])+'；默认参考稳定性问题单列。')
    if scan_v2 and scan_v2.get('deterministic_condition'):
        v=scan_v2['deterministic_condition'];lines.append('- 同一分块候选在双方官方确定性条件下：'+score(v)+' '+count(v)+'；补测'+v['supplementary']['status']+'；条件成果单列。')
    if scan_hybrid:
        lines.append('- 累计积分Triton＋CANN混合候选：默认'+score(scan_hybrid['formal'])+' '+count(scan_hybrid['formal'])+'；补测'+scan_hybrid['supplementary']['status']+'。')
        if scan_hybrid.get('deterministic_condition'):
            v=scan_hybrid['deterministic_condition'];lines.append('- 同一混合候选在双方官方确定性条件下：'+score(v)+' '+count(v)+'；补测'+v['supplementary']['status']+' '+count(v['supplementary'])+'；条件成果单列。')
    lines+=['','CANN包装由Codex实现，厂商实现只作参考；生成优化以KG/CC实际记录为准。',
        'expm1_原合同通过但极小数值补测失败，不能称扩展精度条件下已合格。所有失败版本保留。',
        '核验项：'+str(len(evidence))+'；未结束请求：'+str(len(unknown))+'；核验问题：'+str(len(issues))+'。']
    (BATCH/'交付报告.md').write_text(chr(10).join(lines)+chr(10),encoding='utf8')
    print(json.dumps(dict(regular_accepted=len(accepted),reference_rows=len(rows),comparisons=len(comparisons),evidence=len(evidence),unknown_requests=len(unknown),issues=len(issues))))
if __name__=='__main__':main()
