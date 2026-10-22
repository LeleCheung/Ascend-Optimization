#!/usr/bin/env python3
"""离线核验夜间交付：合同、源码、补充门禁、完整复验与真实采集身份。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_fastest(selected_sha,candidates,expected_cases):
    """按同case真实延迟核对合格候选，不使用相对参考分数排名。"""
    expected=set(expected_cases);timings={}
    if not expected:raise ValueError('选择审计缺少计时合同')
    for candidate in candidates:
        result=candidate['result'];digest=candidate['source_sha256']
        if result.get('status')!='PASSED':raise ValueError('选择审计混入失败候选')
        rows=[r for r in result.get('per_workload',[]) if r.get('phase')=='timing']
        times={r['uuid']:r.get('latency_ms') for r in rows}
        if len(times)!=len(rows) or set(times)!=expected:
            raise ValueError('选择审计的计时case集合不同或重复')
        if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v<=0 for v in times.values()):
            raise ValueError('选择审计存在无效真实延迟')
        if digest in timings and timings[digest]!=times:
            raise ValueError('同源码出现不同正式延迟，不能通过重复测量挑分')
        timings[digest]=times
    if selected_sha not in timings:raise ValueError('入选源码不在合格候选集合')
    latency=lambda values:math.exp(sum(math.log(v) for v in values.values())/len(values))
    measured={digest:latency(values) for digest,values in timings.items()}
    minimum=min(measured.values());selected=measured[selected_sha]
    if selected>minimum and not math.isclose(selected,minimum,rel_tol=1e-12,abs_tol=0.):
        raise ValueError('入选源码不是合格候选中实际耗时最低的一份')
    return dict(compared_sources=len(measured),selected_latency_ms=selected,minimum_latency_ms=minimum)


def verify_main_selection(batch,op,record,state,selected_sha,local):
    candidates=[]
    for key,job in state['jobs'].items():
        if key!='parent:'+op and not key.startswith('validate:'+op+':'):continue
        value=job.get('result') or {}
        if job.get('status')!='completed' or value.get('status')!='passed':continue
        digest=value['source_sha256'];source=local(value['source'])
        if sha(source)!=digest:raise ValueError('备选源码身份改变')
        gates=[read(local(p)) for p in value.get('gates',[])]
        if sorted(g['total'] for g in gates)!=([9,12,42] if op=='selu_' else [18]):
            raise ValueError('备选源码缺少补充门禁')
        for gate in gates:
            if (gate['source_sha256']!=digest or gate['status'].lower()!='passed'
                    or gate['passed']!=gate['total']):raise ValueError('备选源码门禁身份或状态不符')
            if op in ['cumulative_trapezoid','scatter_add','gelu_and_mul']:
                extra=gate.get('timing_input_correctness',{})
                if extra.get('passed')!=record['timing_cases'] or extra.get('total')!=record['timing_cases']:
                    raise ValueError('备选源码缺少性能输入数值检查')
            if op=='uniform' and gate.get('generator_semantics',{}).get('passed')!=3:
                raise ValueError('备选源码缺少随机状态检查')
        path=local(value['result']);result=read(path)
        verify_formal_contract(record,path.parent,result)
        request=read(path.with_name(path.name.replace('.result.json','.request.json')))
        sources=request['implementation']['sources']
        if len(sources)!=1 or sources[0]['content']!=source.read_text():
            raise ValueError('备选源码与正式请求不同')
        candidates.append(dict(source_sha256=digest,result=result))
    expected=[json.loads(line)['name'] for line in (batch/record['catalog_leaf']/'timing.jsonl').read_text().splitlines() if line.strip()]
    return verify_fastest(selected_sha,candidates,expected)


def verify_formal_contract(record,formal,result):
    """核验实际上传bundle和逐case集合，不能仅凭num_passed汇总认定合同一致。"""
    with tarfile.open(formal/'operator-bundle.tar') as bundle:
        members=bundle.getmembers()
        if len(members)!=len(record['contract_sha256']) or any(not m.isfile() for m in members):
            raise ValueError('上传合同文件集合不同')
        actual={m.name:hashlib.sha256(bundle.extractfile(m).read()).hexdigest() for m in members}
        if actual!=record['contract_sha256']:raise ValueError('实际上传合同字节不同')
        cases={}
        for phase in ['correctness','timing']:
            raw=bundle.extractfile(phase+'.jsonl').read().decode('utf-8')
            for line in raw.splitlines():
                if not line.strip():continue
                case=json.loads(line)
                if case['name'] in cases:raise ValueError('合同case身份重复')
                cases[case['name']]=phase
    rows=result.get('per_workload',[])
    observed={r['uuid']:r.get('phase') for r in rows}
    if len(observed)!=len(rows) or observed!=cases or any(r.get('status')!='PASSED' for r in rows):
        raise ValueError('完整结果逐case身份、phase或状态不同')
    inspected=read(formal/'inspect.json')
    # Native /evaluate响应没有benchmark_fingerprint；核对实际请求的内容寻址绑定。
    requests=list(formal.glob('*.request.json'))
    if len(requests)!=1:raise ValueError('无法唯一定位完整评测请求')
    request=read(requests[0]);bundle_id='sha256:'+sha(formal/'operator-bundle.tar')
    if request['binding'].get('bundle_id')!=bundle_id:
        raise ValueError('完整评测请求未绑定已核验上传bundle')
    if result.get('benchmark_fingerprint') is not None and result['benchmark_fingerprint']!=inspected.get('benchmark_fingerprint'):
        raise ValueError('完整结果与上传合同fingerprint不同')


def audit(batch):
    plan=read(batch/'plan.json');state=read(batch/'endurance-state.json')
    if state['status']!='completed':raise ValueError('队列尚未结束')
    delivery=read(batch/'deliverables/manifest.json');checks=[]
    def local(remote):
        prefix='/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/'+batch.name+'/'
        if not remote.startswith(prefix):raise ValueError('证据位于批次之外: '+remote)
        p=batch/remote[len(prefix):];p.resolve().relative_to(batch.resolve());return p
    records={r['operator']:r for r in plan['records']}
    for op,r in records.items():
        for name,digest in r['contract_sha256'].items():
            if sha(batch/r['catalog_leaf']/name)!=digest:raise ValueError('合同身份改变')
            checks.append(dict(kind='contract',operator=op,file=name,sha256=digest))
        if sha(batch/r['seed'])!=r['source_sha256']:raise ValueError('起点源码改变')
    for row in delivery['records']:
        op=row['operator'];selected=row['selected'];digest=selected['source_sha256']
        if sha(local(row['delivered']))!=digest or sha(local(selected['source']))!=digest:
            raise ValueError('交付源码与被测源码不同')
        result_path=local(selected['result']);result=read(result_path)
        if result['status']!='PASSED' or result['num_passed']!=result['num_workloads']:
            raise ValueError('完整评测未通过')
        expected=records[op]['correctness_cases']+records[op]['timing_cases']
        if result['num_workloads']!=expected:raise ValueError('正式用例不完整')
        verify_formal_contract(records[op],result_path.parent,result)
        request=read(result_path.with_name(result_path.name.replace('.result.json','.request.json')))
        sources=request['implementation']['sources']
        if len(sources)!=1 or sources[0]['content']!=local(row['delivered']).read_text():
            raise ValueError('请求源码身份不符')
        gates=[read(local(n)) for n in selected['gates']]
        totals=sorted(g['total'] for g in gates)
        if totals!=([9,12,42] if op=='selu_' else [18]):raise ValueError('补充门禁不完整')
        for g in gates:
            if g['source_sha256']!=digest or g['status'].lower()!='passed' or g['passed']!=g['total']:
                raise ValueError('门禁身份或结果不符')
            if op=='uniform':
                rng=g.get('generator_semantics',{})
                if (rng.get('total')!=3 or rng.get('passed')!=3
                        or [r['case'] for r in rng.get('cases',[])]!=['seed_high_bits','offset_high_bits','state_advance_and_replay']
                        or any(r['status']!='passed' for r in rng.get('cases',[]))):
                    raise ValueError('uniform生成器状态和高位语义检查缺失或失败')
            if op in ['cumulative_trapezoid','scatter_add','gelu_and_mul']:
                extra=g.get('timing_input_correctness',{})
                expected_cases=[json.loads(line)['name'] for line in (batch/records[op]['catalog_leaf']/'timing.jsonl').read_text().splitlines() if line.strip()]
                actual=extra.get('cases',[])
                if (extra.get('total')!=records[op]['timing_cases'] or extra.get('passed')!=extra.get('total')
                        or [r['case'] for r in actual]!=expected_cases or any(r['status']!='passed' for r in actual)):
                    raise ValueError(op+'性能输入的独立数值检查缺失或失败')
                if extra.get('contract_sha256')!={name:records[op]['contract_sha256'][name] for name in ['oracle.py','timing.jsonl']}:
                    raise ValueError(op+'性能输入语义检查使用了不同合同')
        if op=='uniform' and plan.get('uniform_philox_check_required'):
            proof=state['operators'][op].get('philox_check') or {}
            if proof.get('status')!='passed':raise ValueError('修复版缺少Philox逐位核验')
            data=read(local(proof['path']))
            cases=data.get('cases',[])
            manual=[r for r in cases if not r.get('builtin')]
            expected_identities={(seed,offset,tensor) for seed,offset in
                [(0,0),(0,1),(7,0),(1<<31,0),(1<<32,0),(1<<40,1<<40),
                 (0x123456789ABCDEF0,1<<32),(0x7123456789ABCDEF,(1<<32)-16)]
                for tensor in [False,True]}
            identities={(r.get('seed'),r.get('offset'),r.get('from_tensor')) for r in manual}
            if (data.get('source_sha256')!=digest or data.get('status')!='passed'
                    or data.get('passed')!=16 or data.get('total')!=16 or len(manual)!=16
                    or identities!=expected_identities
                    or any(r.get('status')!='passed' or r.get('mismatches')!=0
                           or r.get('echo')!=r.get('echo_expected') for r in manual)):
                raise ValueError('Philox逐位核验身份或结果不符')
            checks.append(dict(kind='philox',operator=op,source_sha256=digest,passed=16,total=16))
        profile=batch/'research'/op/'final/facts.json'
        profile_state='missing'
        if profile.exists():
            facts=read(profile);profile_state=facts['status']
            if facts['source_sha256']!=digest:raise ValueError('最终路径采集源码不同')
            for side,v in facts['sides'].items():
                if v['status']=='completed' and sha(local(v['trace']))!=v['trace_sha256']:
                    raise ValueError('trace字节不同')
            for ir in facts['ir']:
                for artifact in ir['artifacts']:
                    if sha(local(artifact['path']))!=artifact['sha256']:raise ValueError('IR字节不同')
        metrics=batch/'research'/op/'final-metrics/response.json';metrics_state='missing'
        if metrics.exists():
            response=read(metrics);metrics_state=response['status']
            request=read(metrics.with_name('request.json'))
            if request['implementation']['sources'][0]['content']!=local(row['delivered']).read_text():
                raise ValueError('最终metrics源码不同')
            artifacts=metrics.parent/'artifacts/manifest.json'
            if artifacts.exists():
                for item in read(artifacts)['artifacts']:
                    if sha(local(item['path']))!=item['sha256']:raise ValueError('metrics附件身份不同')
        checks.append(dict(kind='delivery',operator=op,source_sha256=digest,formal=expected,
            supplementary=totals,relative_pytorch=selected['geo_mean'],new_source=row['new_source'],
            execution_profile=profile_state,kgs_metrics=metrics_state))
        # 追加采集修复单独验证兼容源码；其有效CC父版不是本修复的可替代对象。
        if not plan.get('parent_batch'):
            selection=verify_main_selection(batch,op,records[op],state,digest,local)
            checks.append(dict(kind='selection',operator=op,source_sha256=digest,**selection))
    if delivery['final_repeats']!=1:raise ValueError('复验次数不符')
    return dict(status='passed',checks=len(checks),records=checks,
        delivered=len(delivery['records']),registered=len(records),
        unclosed=[op for op in records if op not in {r['operator'] for r in delivery['records']}],
        profiling_gaps=[r['operator'] for r in checks if r['kind']=='delivery' and
            (r['execution_profile']!='completed' or r['kgs_metrics']!='completed')],
        scope='离线证据身份核验；性能、是否达标及采集失败分别报告')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    d=audit(a.batch_dir);a.output.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in d.items() if k!='records'},ensure_ascii=False))
