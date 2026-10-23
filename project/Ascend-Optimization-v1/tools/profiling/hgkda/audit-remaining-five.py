#!/usr/bin/env python3
"""离线核验五算子交付身份、全部计时输入数值与实际延迟选版。"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

spec=importlib.util.spec_from_file_location('remaining_audit_base',Path(__file__).with_name('audit-endurance.py'))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
read=base.read
sha=base.sha
EXTRA={'upsample_bilinear2d':24,'_conj_physical':20,'fractional_max_pool3d':18}


def audit(batch):
    plan=read(batch/'plan.json');state=read(batch/'endurance-state.json')
    if state['status']!='completed':raise ValueError('队列尚未结束')
    records={r['operator']:r for r in plan['records']}
    if set(records)!=set(EXTRA)|{'scatter_add','cumulative_trapezoid'}:
        raise ValueError('不是会议剩余五算子合同')
    if plan['physical_card']!=7 or plan['timing_scope']!='walltime':
        raise ValueError('设备或计时口径不同')
    prefix='/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/'+batch.name+'/'
    def local(remote):
        if not remote.startswith(prefix):raise ValueError('证据位于批次外')
        path=batch/remote[len(prefix):];path.resolve().relative_to(batch.resolve());return path
    tools=read(batch/'knowledge/tool-sha256.json')
    supplemental=read(batch/'knowledge/supplemental-tool-sha256.json')
    if supplemental['check-remaining-special.py']!=sha(Path(__file__).with_name('check-remaining-special.py')):
        raise ValueError('额外特殊值检查器字节改变')
    checks=[]
    cases={}
    for op,record in records.items():
        leaf=batch/record['catalog_leaf']
        for name,digest in record['contract_sha256'].items():
            if sha(leaf/name)!=digest:raise ValueError('合同字节改变: '+op+'/'+name)
        if sha(batch/record['seed'])!=record['source_sha256']:raise ValueError('起点源码改变')
        phases={}
        for phase in ['correctness','timing']:
            rows=[json.loads(line) for line in (leaf/(phase+'.jsonl')).read_text(encoding='utf-8').splitlines() if line.strip()]
            if len(rows)!=record[phase+'_cases']:raise ValueError('合同数量不同')
            phases[phase]=[r['name'] for r in rows]
        cases[op]=phases
        checks.append(dict(kind='contract',operator=op,files=len(record['contract_sha256'])))

    def verify_gates(op,value):
        digest=value['source_sha256'];paths=value.get('gates',[])
        if len(paths)!=1:raise ValueError('缺少或混入其他门禁')
        gate=read(local(paths[0]))
        checker='check-remaining-five.py' if op in EXTRA else 'check-endurance.py'
        if (gate.get('operator')!=op or gate.get('source_sha256')!=digest
                or gate.get('checker_sha256')!=tools[checker]
                or gate.get('physical_device')!=7 or gate.get('status')!='passed'
                or gate.get('passed')!=gate.get('total')):
            raise ValueError('门禁身份或状态不同: '+op)
        rows=gate['cases']
        if len(rows)!=gate['total'] or any(r.get('status')!='passed' for r in rows):
            raise ValueError('逐项语义证据缺失')
        if op in EXTRA:
            wanted=sum(len(v) for v in cases[op].values())+EXTRA[op]
            if gate['total']!=wanted:raise ValueError('补充覆盖不完整')
            if gate['contract_sha256']!=records[op]['contract_sha256']:raise ValueError('门禁合同不同')
            for phase,names in cases[op].items():
                actual=[r['case'] for r in rows if r['phase']==phase]
                if actual!=names:raise ValueError('正式输入数值检查不同')
            supplementary=[r['case'] for r in rows if r['phase']=='supplementary']
            if len(set(supplementary))!=EXTRA[op]:raise ValueError('补充用例缺失或重复')
            if any(r.get('repetitions')!=2 or not r.get('stream_switch') for r in rows):
                raise ValueError('缺少重复或换stream检查')
        else:
            if gate['total']!=18:raise ValueError('scatter/scan补充覆盖不完整')
            extra=gate.get('timing_input_correctness',{})
            timing=extra.get('cases',[])
            if (extra.get('passed')!=len(cases[op]['timing']) or extra.get('total')!=len(timing)
                    or [r['case'] for r in timing]!=cases[op]['timing']
                    or any(r.get('status')!='passed' for r in timing)):
                raise ValueError('性能输入数值检查缺失')
            expected={name:records[op]['contract_sha256'][name] for name in ['oracle.py','timing.jsonl']}
            if extra['contract_sha256']!=expected:raise ValueError('性能数值合同不同')
            if any(r.get('repetitions')!=2 or not r.get('stream_switch_checked') for r in rows+timing):
                raise ValueError('缺少重复或换stream检查')

    def verify_candidate(op,value):
        digest=value['source_sha256'];source=local(value['source'])
        if sha(source)!=digest:raise ValueError('候选源码改变')
        verify_gates(op,value)
        path=local(value['result']);result=read(path)
        if (result['status']!='PASSED' or result['num_passed']!=result['num_workloads']
                or result['num_workloads']!=sum(map(len,cases[op].values()))):
            raise ValueError('正式完整结果不通过')
        base.verify_formal_contract(records[op],path.parent,result)
        request=read(path.with_name(path.name.replace('.result.json','.request.json')))
        sources=request['implementation']['sources']
        if len(sources)!=1 or sources[0]['content']!=source.read_text(encoding='utf-8'):
            raise ValueError('正式请求源码不同')
        score=result.get('geo_mean')
        if not isinstance(score,(float,int)) or not math.isfinite(score) or score<=0 or score!=value['geo_mean']:
            raise ValueError('展示分数与正式结果不同')
        return dict(source_sha256=digest,result=result)

    delivery=read(batch/'deliverables/manifest.json')
    if delivery['final_repeats']!=1:raise ValueError('正式复验次数不同')
    delivered=set()
    for row in delivery['records']:
        op=row['operator'];value=row['selected'];digest=value['source_sha256']
        if op in delivered or op not in records:raise ValueError('交付对象重复或未登记')
        delivered.add(op)
        if sha(local(row['delivered']))!=digest:raise ValueError('交付源码不同')
        verify_candidate(op,value)
        pool=[]
        for key,job in state['jobs'].items():
            candidate=job.get('result') or {}
            if (key=='parent:'+op or key.startswith('validate:'+op+':')) and job.get('status')=='completed' and candidate.get('status')=='passed':
                pool.append(verify_candidate(op,candidate))
        selection=base.verify_fastest(digest,pool,cases[op]['timing'])
        facts=batch/'research'/op/'final/facts.json'
        profile='missing'
        if facts.exists():
            data=read(facts);profile=data['status']
            if data['source_sha256']!=digest:raise ValueError('最终路径采集源码不同')
            for side in data['sides'].values():
                if side['status']=='completed' and sha(local(side['trace']))!=side['trace_sha256']:
                    raise ValueError('最终trace字节不同')
            for entry in data['ir']:
                for artifact in entry['artifacts']:
                    if sha(local(artifact['path']))!=artifact['sha256']:raise ValueError('最终IR字节不同')
        checks.append(dict(kind='delivery',operator=op,source_sha256=digest,score=value['geo_mean'],
            formal=sum(map(len,cases[op].values())),meets_target=value['geo_mean']>.8,
            execution_profile=profile,selection=selection,
            reference_review_required=op in ['fractional_max_pool3d','_conj_physical']))
        if op=='fractional_max_pool3d':
            extra=read(batch/'results'/op/'final-special.json')
            identities={(r['dtype'],r['ndim'],r['pattern']) for r in extra.get('cases',[])}
            expected={(dtype,ndim,pattern) for dtype in ['torch.float16','torch.bfloat16','torch.float32']
                for ndim in [4,5] for pattern in ['negative_inf','mixed_inf','multiple_nan','ties','strided_samples']}
            checker=Path(__file__).with_name('check-remaining-special.py')
            if (extra.get('source_sha256')!=digest or extra.get('checker_sha256')!=sha(checker)
                    or extra.get('status')!='passed' or extra.get('passed')!=30 or extra.get('total')!=30
                    or len(extra.get('cases',[]))!=30 or identities!=expected
                    or any(r['status']!='passed' or r.get('repetitions')!=2 or not r.get('stream_switch') for r in extra['cases'])):
                raise ValueError('池化特殊值、采样stride的补充验收缺失或失败')
            checks.append(dict(kind='fractional_special',operator=op,total=30,source_sha256=digest))
        if op=='upsample_bilinear2d':
            extra=read(batch/'results'/op/'final-special.json')
            identities={(r['dtype'],r['case']) for r in extra.get('cases',[])}
            expected={(dtype,index) for dtype in ['torch.float16','torch.bfloat16','torch.float32'] for index in range(12)}
            if (extra.get('source_sha256')!=digest or extra.get('checker_sha256')!=sha(Path(__file__).with_name('check-remaining-special.py'))
                    or extra.get('status')!='passed' or extra.get('passed')!=36 or extra.get('total')!=36
                    or len(extra.get('cases',[]))!=36 or identities!=expected
                    or any(r['status']!='passed' or r.get('repetitions')!=2 or not r.get('stream_switch') for r in extra['cases'])):
                raise ValueError('上采样显式scale与布局补充验收缺失或失败')
            checks.append(dict(kind='upsample_special',operator=op,total=36,source_sha256=digest))
        if op=='_conj_physical':
            path=batch/'research'/op/'final-complex/facts.json'
            data=read(path)
            if data['source_sha256']!=digest or data['case']['inputs']['spec']['dtype']!='complex64':
                raise ValueError('缺少同源码复数参考执行路径核对')
            for side in data['sides'].values():
                if side['status']=='completed' and sha(local(side['trace']))!=side['trace_sha256']:
                    raise ValueError('复数执行路径trace字节不同')
            checks.append(dict(kind='conj_complex_path',operator=op,status=data['status'],source_sha256=digest))
    missing_best={op for op,d in state['operators'].items() if d.get('best')}-delivered
    if missing_best:raise ValueError('有有效最佳但未列入交付')
    return dict(status='passed',registered=len(records),delivered=len(delivered),checks=len(checks),records=checks,
        unclosed=sorted(set(records)-delivered),
        under_target=[r['operator'] for r in checks if r['kind']=='delivery' and not r['meets_target']],
        scope='离线证据核验；参考CPU回退需人工审阅trace并单列，不将分数自动认定为原生NPU优势')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=audit(args.batch_dir)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False))
