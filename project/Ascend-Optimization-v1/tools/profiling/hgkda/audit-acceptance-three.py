#!/usr/bin/env python3
"""核对剩余三项的合同、源码、两类参考、补充门禁和默认扫描反例。"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(batch):
    checks=[]
    def check(condition,label):
        if not condition:raise ValueError(label)
        checks.append(label)
    def resolve(name):
        marker='/acceptance-three-20261026/'
        if marker in str(name):return batch/str(name).split(marker,1)[1]
        return batch/name
    plan=read(batch/'plan.json')
    for row in plan['records']:
        for name,digest in row['contract_sha256'].items():
            check(sha(batch/row['catalog_leaf']/name)==digest,row['operator']+':原合同:'+name)
    baseline=read(batch/'baseline-verified.json')
    check(baseline['status']=='verified' and baseline['flaggems_files_checked']==6352,'固定FlagGems源码核验')
    check(baseline['service_software']['compiler_version']=='0.6.0+ascend.gitc286cba6'
        and baseline['service_software']['runtime_version']=='9.0.0','固定编译器与CANN')
    operators={}
    for op,count in [('fractional_max_pool3d',9),('_conj_physical',14)]:
        source=batch/'deliverables'/op/'main.py'
        digest=sha(source)
        out=batch/'validation-v2'/op
        conditions=read(out/'conditions.json')
        check(conditions['source_sha256']==digest and conditions['physical_device']==7 and not conditions['deterministic'],op+':条件与源码')
        semantic=read(out/'semantics.json')
        check(semantic['status']=='passed' and semantic['passed']==semantic['total']
            and semantic['source_sha256']==digest,op+':全部补充数值布局与stream')
        pair=read(out/'paired-numerics.json')
        check(pair['status']=='passed' and pair['total']==pair['passed']==count
            and pair['source_sha256']==digest,op+':所有正式输入两类参考逐值一致')
        check(all(x['repetitions']==2 and x['stream_switch'] for x in pair['cases']),op+':重复与换stream')
        if op=='fractional_max_pool3d':
            special=read(out/'special.json')
            check(special['status']=='passed' and special['passed']==special['total']==30
                and special['source_sha256']==digest,op+':30项特殊值')
        leaf=batch/'catalog/ops'/op
        supplemental=batch/'npu-decomposition-catalog-v2/ops'/op
        for p in leaf.iterdir():
            if p.is_file() and p.name!='oracle.py':
                check(p.read_bytes()==(supplemental/p.name).read_bytes(),op+':补充组保留合同:'+p.name)
        check(sha(supplemental/'oracle.py')==conditions['reference_sha256'],op+':补充参考源码')
        groups={}
        for label in ['default-reference','npu-decomposition-reference']:
            result=read(out/(label+'.result.json'))
            request=read(out/(label+'.request.json'))
            preflight=read(out/(label+'.preflight.json'))
            check(preflight['status']=='PASSED' and not preflight['is_hack'],op+':'+label+':准入')
            check(result['status']=='PASSED' and result['num_passed']==result['num_workloads']==count
                and not result['is_hack'],op+':'+label+':完整通过')
            sources=request['implementation']['sources']
            check(len(sources)==1 and hashlib.sha256(sources[0]['content'].encode()).hexdigest()==digest,op+':'+label+':实际源码')
            for key,file in [('correctness_workloads','correctness.jsonl'),('timing_workloads','timing.jsonl')]:
                expected=[readline for readline in (json.loads(l) for l in (leaf/file).read_text().splitlines())]
                actual=request[key]
                def subset(old,new):
                    return all(k in new and subset(v,new[k]) for k,v in old.items()) if isinstance(old,dict) else old==new
                check(len(expected)==len(actual) and all(subset(a,b) for a,b in zip(expected,actual)),op+':'+label+':'+key)
            rows=[x for x in result['per_workload'] if x['phase']=='timing']
            check(len(rows)==6 and all(x['latency_ms']>0 and x['reference_latency_ms']>0 for x in rows),op+':'+label+':完整实际延迟')
            score=math.exp(sum(math.log(x['reference_latency_ms']/x['latency_ms']) for x in rows)/len(rows))
            check(math.isclose(score,result['geo_mean'],rel_tol=1e-7),op+':'+label+':重算几何加速比')
            groups[label]=dict(score=score,result=str((out/(label+'.result.json')).relative_to(batch)),
                per_case=[{k:x[k] for k in ['uuid','latency_ms','reference_latency_ms']} for x in rows])
        facts=read(out/'npu-paired-path/facts.json')
        check(facts['status']=='completed' and facts['physical_device']==7 and facts['source_sha256']==digest,op+':同源码成对采集')
        for side,row in facts['sides'].items():
            check(row['status']=='completed' and sha(resolve(row['trace']))==row['trace_sha256'],op+':原始trace:'+side)
        check(bool(facts['ir']),op+':存在实际IR')
        for item in facts['ir']:
            for artifact in item['artifacts']:
                check(sha(resolve(artifact['path']))==artifact['sha256'],op+':IR:'+artifact['kind'])
        operators[op]=dict(source=str(source.relative_to(batch)),source_sha256=digest,formal=count,
            semantic_total=semantic['total'],reference_groups=groups,status='supplemental_npu_comparison_passed',
            default_acceptance='仍为CPU回退参考，不能计为正常NPU官方对照达标',
            contribution='Codex连续输出分块改写；KG历史候选为父版' if op=='fractional_max_pool3d' else '历史KG/Codex实现复用；本批DMA候选未采纳')
    default=read(batch/'research/reference-reproducer/reference-default.json')
    deterministic=read(batch/'research/reference-reproducer/reference-deterministic.json')
    check(default['status']=='reference_inconsistent' and default['empty_intersections']>0 and not default['deterministic'],'默认扫描参考反例存在')
    check(all(x['maximum_lower_bound']>x['minimum_upper_bound'] and x['gap']>0 for x in default['counterexamples']),'默认参考允许区间无交集')
    check(deterministic['deterministic'] and deterministic['bitwise_stable'] and deterministic['empty_intersections']==0,'官方确定性重复稳定')
    check(sha(batch/'reproducer-input.bin')==default['input_sha256']==deterministic['input_sha256'],'反例固定字节身份')
    rows=read(batch/'research/cumulative/results.json')['rows']
    check(any(r['label']=='candidate_v2' and not r['deterministic'] and not r['bitwise_stable'] for r in rows),'CANN扫描V2不能解决默认稳定性')
    scan=batch/'deliverables/cumulative_trapezoid/main.py'
    check(sha(scan)=='aa2c6cce2432600b5171a6f1848e1b5707f8464890d6d9a1114a69c0182e68a6','保留已完整验证确定性扫描源码')
    operators['cumulative_trapezoid']=dict(source=str(scan.relative_to(batch)),source_sha256=sha(scan),
        status='default_reference_blocked',deterministic_inherited_score=0.8124123961740622,
        inherited_formal=33,default_reference_counterexamples=default['empty_intersections'],
        reference_reproducer='research/reference-reproducer/reference-default.json',
        note='本批不重复正式性能测试；确定性结果来自two-closure-20261024，不是本批新优化')
    parent=read(batch/'validation-v2/fractional_max_pool3d/parent-comparison/result.json')
    comparison=read(batch/'validation-v2/fractional_max_pool3d/parent-comparison/comparison.json')
    check(parent['status']=='PASSED' and parent['num_passed']==parent['num_workloads']==9 and not parent['is_hack'],'池化父版同批完整评测')
    final=read(batch/'validation-v2/fractional_max_pool3d/default-reference.result.json')
    a={r['uuid']:r['latency_ms'] for r in parent['per_workload'] if r['phase']=='timing'}
    b={r['uuid']:r['latency_ms'] for r in final['per_workload'] if r['phase']=='timing'}
    check(set(a)==set(b) and len(a)==6,'池化父版与最终相同计时case')
    improvement=math.exp(sum(math.log(a[k]/b[k]) for k in a)/len(a))
    check(comparison['status']=='passed' and math.isclose(improvement,comparison['actual_speedup'],rel_tol=1e-8),'池化实际延迟收益重算')
    check(comparison['final_source_sha256']==operators['fractional_max_pool3d']['source_sha256'],'池化收益对应最终源码')
    operators['fractional_max_pool3d']['actual_speedup_over_parent']=improvement
    for mode in ['default','deterministic']:
        binary=batch/('research/reference-reproducer/reference-'+mode+'.outputs.npy')
        check(binary.exists() and binary.stat().st_size>16*255*8*8,'参考16次完整输出:'+mode)
    for name in ['validate-v2-fractional_max_pool3d','validate-v2-_conj_physical','reference-reproducer-default','reference-reproducer-deterministic','pool-parent-comparison']:
        state=read(batch/'stages'/name/'state.json')
        response=read(batch/'stages'/name/'response.json')
        check(state['status']==response['status']=='SUCCEEDED' and state['job_id']==response['job_id'],'KGS作业回执:'+name)
    return dict(status='passed',checks=len(checks),checked=checks,operators=operators,
        acceptance=dict(default_passed=7,conditional_scan=1,supplemental_npu_comparison=2,all_ten_default_passed=False),
        caveat='NPU分解是自建多原语补充参考，带启动/同步成本；不能冒充官方原生单kernel')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    a=parser.parse_args()
    assert not a.output.exists()
    result=audit(a.batch_dir)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+chr(10),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],checks=result['checks'],acceptance=result['acceptance']),ensure_ascii=False))
