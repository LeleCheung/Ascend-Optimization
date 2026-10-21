#!/usr/bin/env python3
"""单列额外修复的谱系、语义和复验，不改写原始三组的结论。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics

spec=importlib.util.spec_from_file_location('night_report',Path(__file__).with_name('report-nightly.py'))
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_identity(data,source_sha,checker_sha,operator='value_selecting_reduction_backward'):
    return report.valid_semantics(data,operator,source_sha,checker_sha)


def verify(batch):
    plan=report.read(batch/'plan.json')
    if plan.get('execution_mode') not in ['semantic_repair','moe_semantic_repair']:raise ValueError('不是单列的语义修复批次')
    parent=batch.parent/plan['parent_batch'];parent.resolve().relative_to(batch.parent.resolve())
    records=plan['records']
    if len(records)!=1:raise ValueError('修复批次必须只含一个算子')
    record=records[0];op=record['operator']
    if op!=('moe_sum_reduce' if plan['execution_mode']=='moe_semantic_repair' else 'value_selecting_reduction_backward'):
        raise ValueError('修复模式与算子不同')
    original=next(r for r in report.read(parent/'plan.json')['records'] if r['operator']==op)
    source=batch/record['seed'];source.resolve().relative_to(batch.resolve())
    if sha(source)!=record['source_sha256']:raise ValueError('修复源码与plan不一致')
    if record['contract_sha256']!=original['contract_sha256']:raise ValueError('修复改变了原始合同')
    if record['upstream_commit']!=original['upstream_commit']:raise ValueError('修复改变了固定版本')
    leaf=batch/record['catalog_leaf'];leaf.resolve().relative_to(batch.resolve())
    for name,digest in record['contract_sha256'].items():
        if sha(leaf/name)!=digest:raise ValueError('修复合同字节不匹配')
    arm=plan['parent_arm']
    if arm not in ['kg-no-profile','kg-profile','HG-KDA']:raise ValueError('未支持的父组')
    native=parent/'results'/op/arm/'native-evidence'
    entries=report.read(native/'archive-manifest.json')['files']
    matches=[]
    for entry in entries:
        match=re.fullmatch(r'\.kernelgen/evals/round-(\d{4})/main\.py',entry['source_path'])
        if (match and entry['sha256']==record['parent_generated_source_sha256'] and
            (plan.get('parent_round') is None or int(match[1])==plan['parent_round'])):
            matches.append((entry,int(match[1])))
    if len(matches)!=1:raise ValueError('父轮次源码身份没有唯一对应的原始归档')
    entry,parent_round=matches[0];artifact=native/entry['archived_path'];artifact.resolve().relative_to(native.resolve())
    if sha(artifact)!=entry['sha256'] or not source.read_bytes().startswith(artifact.read_bytes()):
        raise ValueError('父源码被修改或修复未保留登记的原始前缀')
    data={'batch':batch.name,'operator':op,'parent_batch':parent.name,'parent_arm':arm,
          'parent_round':parent_round,'parent_source_sha256':entry['sha256'],
          'source_sha256':record['source_sha256'],'plan_sha256':sha(batch/'plan.json'),
          'status':'pending','additional_budget':plan['additional_budget'],
          'contribution':plan['contribution'],'same_budget_generated_result':False}
    semantic_failure=plan.get('semantic_failure_provenance')
    if semantic_failure:
        observed=batch/semantic_failure['snapshot'];observed.resolve().relative_to(batch.resolve())
        original_failure=parent/semantic_failure['parent_path'];original_failure.resolve().relative_to(parent.resolve())
        failure=report.read(observed)
        if (sha(observed)!=semantic_failure['sha256'] or sha(original_failure)!=sha(observed)
                or failure.get('source_sha256')!=record['parent_generated_source_sha256']
                or failure.get('operator')!=op or failure.get('status')!='failed'):
            raise ValueError('修复未对应原始同源码语义失败')
        data['semantic_failure_provenance_verified']=True
    diagnostic=plan.get('diagnostic_provenance')
    if diagnostic:
        analysis_path=batch/diagnostic['analysis_path'];analysis_path.resolve().relative_to(batch.resolve())
        if sha(analysis_path)!=diagnostic['analysis_sha256']:raise ValueError('采用的诊断快照身份不同')
        matches=[e for e in entries if e['source_path']==diagnostic['parent_analysis_path'] and
                 e['sha256']==diagnostic['analysis_sha256']]
        if len(matches)!=1:raise ValueError('诊断未对应父实验的唯一原始归档')
        analysis=report.read(analysis_path)
        verified=report.check_profile_evidence(parent/'results'/op/arm)['verified_completed_analyses']
        if (analysis.get('status')!='completed' or analysis.get('round_num')!=parent_round or
            analysis.get('solution_sha256')!=diagnostic['analysis_solution_sha256'] or
            not any(v['round']==parent_round and v['solution_sha256']==analysis['solution_sha256'] for v in verified)):
            raise ValueError('诊断缺少同轮次真实采集与分析证据')
        data['profiling_guidance_verified']=True
        data['analysis_sha256']=diagnostic['analysis_sha256']
    state=report.read(batch/'queue-state.json') if (batch/'queue-state.json').exists() else {}
    if state.get('plan_sha256') and state['plan_sha256']!=data['plan_sha256']:
        raise ValueError('运行时plan身份改变')
    root=batch/'results'/op/'codex-stride-repair'
    sem=root/'supplementary-semantics.json'
    if not sem.exists():return data
    value=report.read(sem)
    data['supplementary_semantics']={k:value.get(k) for k in ['status','passed','total','source_sha256','checker_sha256']}
    checker_sha=sha(Path(__file__).with_name('check-nightly-semantics.py'))
    if plan.get('checker_sha256',checker_sha)!=checker_sha:
        raise ValueError('冻结检查器身份变化')
    if not semantic_identity(value,record['source_sha256'],checker_sha,op):
        data.update(status='invalid',reason='%d项同身份补充语义尚未通过'%(36 if op=='moe_sum_reduce' else 48));return data
    if plan.get('additional_semantics'):
        extra=plan['additional_semantics'];script=report.PROJECT/extra['script']
        if sha(script)!=extra['script_sha256']:raise ValueError('额外检查器与冻结plan不同')
        dense=root/'dense-tile-semantics.json'
        if not dense.exists():return data
        spec=importlib.util.spec_from_file_location('semantic_repair_checks',Path(__file__).with_name('run-semantic-repair.py'))
        checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)
        dense_result=report.read(dense)
        if not checks.valid_dense_semantics(dense_result,record['source_sha256'],extra['script_sha256']):
            data.update(status='invalid',reason='18项跨列块补测尚未同身份通过');return data
        data['dense_tiles_verified']=True
    loaded=[]
    for i in [1,2]:
        path=root/'independent'/('codex-stride-repair-%d.result.json'%i)
        if not path.exists():return data
        if report.read(path).get('status')!='PASSED':
            data.update(status='invalid',reason='完整KGS复验失败');return data
        result=report.verify(path,record,leaf)
        if result['source_sha256']!=record['source_sha256']:raise ValueError('复验未测试登记的修复源码')
        loaded.append(result)
    data['scores']=[statistics.geometric_mean(r['reference_latency_ms']/r['latency_ms'] for r in v['timing'].values()) for v in loaded]
    data['workloads']=len(loaded[0]['rows']);data['status']='verified'
    data['result_sha256']=[v['result_sha256'] for v in loaded]
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args();rows=[verify(path.resolve()) for path in args.batch_dir]
    value={'all_repairs_verified':all(r['status']=='verified' for r in rows),'repairs':rows,
           'scope':'额外Codex修复单列；不替换原始KG/HG-KDA相同预算实验，也不修改原报告严格门禁'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())
    print(json.dumps(value,ensure_ascii=False))
    if args.require_complete and not value['all_repairs_verified']:raise SystemExit(1)


if __name__=='__main__':main()
