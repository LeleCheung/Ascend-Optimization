#!/usr/bin/env python3
"""离线核验夜间交付：合同、源码、补充门禁、完整复验与真实采集身份。"""
import argparse
import hashlib
import json
from pathlib import Path


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


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
    if delivery['final_repeats']!=1:raise ValueError('复验次数不符')
    return dict(status='passed',checks=len(checks),records=checks,
        delivered=len(delivery['records']),registered=len(records),
        unclosed=[op for op in records if op not in {r['operator'] for r in delivery['records']}],
        scope='离线证据身份核验；性能、是否达标及采集失败分别报告')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    d=audit(a.batch_dir);a.output.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in d.items() if k!='records'},ensure_ascii=False))
