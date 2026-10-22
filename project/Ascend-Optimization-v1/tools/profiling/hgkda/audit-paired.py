#!/usr/bin/env python3
"""离线验收成对批次：交付、完整用例、采集、IR和实际角色加载身份。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def load_module(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def local(batch,remote):
    prefix='/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/'+batch.name+'/'
    if not remote.startswith(prefix):raise ValueError('证据位于批次之外')
    p=batch/remote[len(prefix):]
    p.resolve().relative_to(batch.resolve())
    return p


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tag', default='final-v6', help='对应completion-state与deliverables的最终版本名')
    a=p.parse_args();b=a.batch_dir
    if not a.tag.replace('-', '').isalnum():raise ValueError('非法最终版本名')
    if a.output.exists():raise ValueError('验收记录已存在')
    tools=Path(__file__).resolve().parent
    exporter=load_module('paired_audit_exporter',tools/'export-paired-deliverables.py')
    compare=load_module('paired_audit_compare',tools/'skills/hgkda/scripts/compare_opprof.py')
    state=json.loads((b/('completion-state-'+a.tag+'.json')).read_text())
    if state['status']!='completed':raise ValueError('最终队列未结束')
    delivery=json.loads((b/('deliverables-'+a.tag)/'manifest.json').read_text())
    if not delivery['dense_gate_required'] or delivery['final_repeats']!=1:raise ValueError('最终门禁配置不符')
    checks=[]
    def checked(kind,**data):checks.append(dict(kind=kind,**data))
    plan=json.loads((b/'plan.json').read_text())
    for record in plan['records']:
        for name,digest in record['contract_sha256'].items():
            if exporter.sha(b/record['catalog_leaf']/name)!=digest:raise ValueError('合同已改变')
            checked('contract',operator=record['operator'],file=name,sha256=digest)
    for row in delivery['records']:
        op=row['operator'];selected=row['selected']
        verified=exporter.read_arm(b,op,selected['arm'],require_dense=True)
        if not verified or verified['source_sha256']!=selected['source_sha256']:
            raise ValueError('交付与完整评测身份不符')
        if (b/'require-range-gate.json').exists() and verified['range_cases'] != 9:
            raise ValueError('本批交付缺少9项数值范围门禁')
        delivered=local(b,row['delivered'])
        if exporter.sha(delivered)!=verified['source_sha256']:raise ValueError('交付字节不同')
        checked('delivery',operator=op,source_sha256=verified['source_sha256'],geo_mean=verified['geo_mean'],
                formal=verified['workloads'],semantics=verified['semantics_cases'],dense=verified['dense_cases'],
                ranges=verified['range_cases'])
        proof=state['results'][op]
        if proof['source_sha256']!=verified['source_sha256']:raise ValueError('采集源码不同')
        response_path=local(b,proof['kgs_profile']);response=json.loads(response_path.read_text())
        request=json.loads(response_path.with_name('request.json').read_text())
        source=delivered.read_text()
        if (response['status']!='completed' or not response.get('metrics',{}).get('ops')
            or request['implementation']['sources'][0]['content']!=source):
            raise ValueError('KGS profiling未完成或源不同')
        manifest=json.loads((response_path.parent/'artifacts/manifest.json').read_text())
        for item in manifest['artifacts']:
            file=local(b,item['path'])
            if exporter.sha(file)!=item['sha256'] or file.stat().st_size!=item['size_bytes']:
                raise ValueError('KGS附件身份不同')
            checked('kgs_artifact',operator=op,path=str(file),sha256=item['sha256'])
        if len(proof['pairs'])!=2:raise ValueError('最终需要小/大两组成对采集')
        for paired in proof['pairs']:
            directory=local(b,paired).parent
            facts=compare.compare(directory)
            if facts['spec']['source_sha256']!=verified['source_sha256']:raise ValueError('成对源码不同')
            for side in ['reference','candidate']:
                meta=facts[side]['metadata']
                if not meta.get('completion_marker') or meta.get('returncode')!=0:
                    raise ValueError('成对采集缺少完成标记')
                if meta['driver_sha256']!=exporter.sha(directory/'driver.py'):raise ValueError('driver身份不同')
            checked('paired',operator=op,case_id=facts['spec']['case_id'],
                    reference_us=facts['reference']['basic']['Task Duration(us)'],
                    candidate_us=facts['candidate']['basic']['Task Duration(us)'])
        for label in ['seed','final']:
            ir=response_path.parent.parent/(op+'-'+label+'-ir')
            manifest=json.loads((ir/'manifest.json').read_text())
            if label=='final' and manifest['source_sha256']!=verified['source_sha256']:
                raise ValueError('IR源码不同')
            entries=manifest['entries']
            if not entries:raise ValueError('IR为空')
            for entry in entries:
                kinds={item['kind'] for item in entry['artifacts']}
                if not {'ttir','npubin'}.issubset(kinds):raise ValueError('缺少实际IR或二进制')
                for item in entry['artifacts']:
                    file=local(b,item['path'])
                    if exporter.sha(file)!=item['sha256']:raise ValueError('IR字节不同')
                    checked('ir',operator=op,label=label,format=item['kind'],sha256=item['sha256'])
    for op in ['log10_','selu_']:
        d=b/'results'/op/'paired-HG-KDA'
        roles=json.loads((d/'role-load-evidence.json').read_text())
        if not roles['roles']:raise ValueError('没有真实角色载入证据')
        for role in roles['roles']:
            file=d/role['archived']
            if exporter.sha(file)!=role['sha256'] or 'hgkda-paired-v1' not in file.read_text():
                raise ValueError('实际角色或成对知识身份不符')
        archive=json.loads((d/'native-evidence/archive-manifest.json').read_text())
        analyses=[entry for entry in archive['files'] if entry['original_archive_path'].startswith('profile-analysis/')]
        if not analyses:raise ValueError('无实际结构化分析记录')
        for entry in analyses:
            file=d/'native-evidence'/entry['archived_path']
            if exporter.sha(file)!=entry['sha256']:raise ValueError('分析记录身份不同')
        checked('analysis_execution',operator=op,records=len(analyses),
                delegated_role_calls=len(archive['model_role_invocations']),
                note='无独立委派时仅认定主agent保存结构化分析，不冒称独立子agent调用')
    output=dict(status='passed',checks=len(checks),records=checks,
        all_meet_08=delivery['all_meet_08'],scope='成对批次原始证据离线核验；性能是否达标单独判断')
    a.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k!='records'},ensure_ascii=False))


if __name__=='__main__':main()
