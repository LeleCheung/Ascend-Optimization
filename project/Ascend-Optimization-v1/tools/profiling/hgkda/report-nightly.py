#!/usr/bin/env python3
"""按冻结合同核验夜间结果，生成中文体检表和四组逐case对照。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile

PROJECT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('comparison',PROJECT/'tools/evaluation/compare-operator-versions.py')
comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def valid_semantics(data,operator,source_sha,checker_sha):
    expected={'value_selecting_reduction_backward':48,'moe_sum_reduce':36}[operator]
    cases=data.get('cases',[])
    expected_cases={(dtype,i) for dtype in ['torch.float16','torch.bfloat16','torch.float32']
                    for i in range(expected//3)}
    return bool(data.get('status')=='passed' and data.get('operator')==operator and
                data.get('source_sha256')==source_sha and data.get('checker_sha256')==checker_sha and
                data.get('physical_device')==7 and data.get('logical_device')=='npu:0' and
                data.get('passed')==data.get('total')==expected and len(cases)==expected and
                {(r.get('dtype'),r.get('case')) for r in cases}==expected_cases and
                all(r.get('status')=='passed' for r in cases))


def guided_rounds(rounds,verified_analyses):
    """不同KG入口会登记analyzer或profile_next_experiment；均须对应已核验父分析。"""
    parents={row['round'] for row in verified_analyses}
    return [row for row in rounds
            if row['source'].get('origin') in ['profile_next_experiment','analyzer']
            and row['source'].get('parent_round_num') in parents
            and row['round']>row['source']['parent_round_num']]


def check_profile_evidence(root):
    manifest=root/'native-evidence/archive-manifest.json'
    entries=read(manifest)['files'] if manifest.exists() else []
    analyses=[];profiles={};all_profiles={};identities=set();invocations=[];loaded_roles=[];debug_jobs=[];rounds=[]
    for entry in entries:
        artifact=root/'native-evidence'/entry['archived_path']
        artifact.resolve().relative_to((root/'native-evidence').resolve())
        if hashlib.sha256(artifact.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('原生证据SHA不匹配: '+str(artifact))
        original=entry['source_path']
        if original.endswith('/.ledger.json') or original=='.ledger.json':
            for row in read(artifact).get('rounds',[]):
                rounds.append({'round':row['round_num'],'source':row.get('plan',{}).get('source',{}),
                               'source_sha256':row.get('solution',{}).get('sha256'),
                               'status':row.get('evaluation',{}).get('status')})
        elif 'debug-jobs/' in original and original.endswith('/status.json'):
            job=read(artifact);debug_jobs.append({k:job.get(k) for k in ['job_id','status','started_at','finished_at','exit_code']})
        elif original.endswith(('/kernel-profile-analyzer.md','/kernel-knowledge-profile-analyzer.md')):
            if 'HG-KDA 角色扩展' in artifact.read_text(encoding='utf-8'):
                loaded_roles.append({'archived':'native-evidence/'+entry['archived_path'],
                                     'sha256':entry['sha256'],'path':original})
        elif original.endswith('/run-events.jsonl'):
            for line in artifact.read_text(encoding='utf-8').splitlines():
                event=json.loads(line)
                if (event.get('event_type')=='MODEL_INVOCATION_COMPLETED' and
                    event.get('data',{}).get('agent') in ['kernel-profile-analyzer','kernel-knowledge-profile-analyzer']):
                    invocations.append(event)
        elif 'profile-analysis/' in original and original.endswith('.json'):
            analyses.append(read(artifact))
        elif 'profiles/' in original and original.endswith('/manifest.json'):
            data=read(artifact)
            if data.get('result',{}).get('profile_id'):
                all_profiles[data['result']['profile_id']]=data
            if data.get('result',{}).get('status')=='completed' and data['result'].get('artifacts'):
                profiles[data['result']['profile_id']]=data
        elif '/evals/' in original and original.endswith('/identity.json'):
            identities.add(read(artifact)['solution_sha256'])
    verified=[]
    for analysis in analyses:
        if analysis.get('status')!='completed' or not analysis.get('findings') or not analysis.get('next_experiment'):
            continue
        sha=analysis['solution_sha256']
        if sha not in identities:raise ValueError('profiling分析未对应归档的评测轮次')
        matched=[]
        for workload in analysis.get('profiled_workloads',[]):
            if workload.get('status')!='completed':continue
            for profile_id in workload.get('profile_ids',[]):
                profile=all_profiles.get(profile_id)
                if profile is None:raise ValueError('分析引用的profiling回执未归档: '+profile_id)
                if profile['request']['identity']['solution_sha256']!=sha:
                    raise ValueError('profiling与分析的候选身份不一致')
                if profile['request']['case_id']!=workload['uuid']:
                    raise ValueError('profiling与分析的case不一致')
                if profile_id in profiles:matched.append(profile_id)
        if matched:verified.append({'round':analysis['round_num'],'solution_sha256':sha,'profile_ids':matched})
    return {'profile_analysis_files':len(analyses),'successful_profiles':len(profiles),'analyzer_invocations':invocations,
            'failed_profiles':[{'profile_id':key,'status':value['result'].get('status'),
                                'case_id':value['request'].get('case_id'),'error':value['result'].get('error')}
                               for key,value in all_profiles.items() if value['result'].get('status')=='failed'],
            'verified_completed_analyses':verified,'archived_hgkda_roles':loaded_roles,
            'formal_rounds':rounds,'debug_jobs':debug_jobs,
            'profile_guided_rounds':guided_rounds(rounds,verified),
            'analyzer_delegations':read(manifest).get('model_role_invocations',[]) if manifest.exists() else []}


def verify(path,record,leaf):
    for name,digest in record['contract_sha256'].items():
        if hashlib.sha256((leaf/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('冻结合同文件SHA不匹配: '+name)
    loaded=comparison.load_result(path)
    expected_software=read(PROJECT.parents[1]/'manifests/hgkda-stage2.lock.json')['service_software']
    service=read(path.parent/'server-status.json')
    if any(service.get('software',{}).get(k)!=v for k,v in expected_software.items()):
        raise ValueError('评测时FlagTree/CANN等软件版本发生变化')
    if service.get('target',{}).get('backend')!='ascend':raise ValueError('评测并非Ascend后端')
    bundle=path.parent/'operator-bundle.tar'
    if loaded['binding'].get('bundle_id')!='sha256:'+hashlib.sha256(bundle.read_bytes()).hexdigest():
        raise ValueError('请求未绑定归档合同bundle')
    with tarfile.open(bundle) as archive:
        observed={}
        for member in archive:
            if not member.isfile() or member.name in observed:raise ValueError('合同归档含重复或非常规文件')
            observed[member.name]=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    if observed!=record['contract_sha256']:raise ValueError('实际评测bundle与冻结合同不同')
    expected={row['name'] for phase in ['correctness','timing'] for row in
              (json.loads(line) for line in (leaf/(phase+'.jsonl')).read_text(encoding='utf-8').splitlines() if line)}
    if set(loaded['rows'])!=expected:
        raise ValueError('完整用例与冻结合同不同: '+str(path))
    if loaded['binding']['definition']!=record['operator'] or loaded['timing_scope']!='walltime':
        raise ValueError('算子或Native计时口径不匹配')
    return loaded


def baseline_location(batch,record,issue):
    original=batch/'results'/record['operator']/'FlagGems/FlagGems-1.result.json'
    if not issue:return record,batch/record['catalog_leaf'],original,True
    retry=batch.parent/issue['retest_batch'];retry.resolve().relative_to(batch.parent.resolve())
    plan=retry/'plan.json'
    if hashlib.sha256(plan.read_bytes()).hexdigest()!=issue['retest_plan_sha256']:
        raise ValueError('接口补测plan身份变化')
    observed=next(r for r in read(plan)['records'] if r['operator']==record['operator'])
    for key in ['upstream_commit','upstream_source_sha256']:
        if observed[key]!=record[key]:raise ValueError('接口补测改变固定上游来源')
    leaf=retry/observed['catalog_leaf']
    for name in ['definition.json','correctness.jsonl','timing.jsonl']:
        if (leaf/name).read_bytes()!=(batch/record['catalog_leaf']/name).read_bytes():
            raise ValueError('接口补测改变ABI或用例，须另设实验')
    result=retry/'results'/record['operator']/'FlagGems/FlagGems-1.result.json'
    return (observed,leaf,result,True) if result.exists() else (record,batch/record['catalog_leaf'],original,False)


def build(batch,require_complete=False):
    plan=read(batch/'plan.json');state=read(batch/'queue-state.json')
    reference_audits=read(batch/'reference-execution-audit.json') if (batch/'reference-execution-audit.json').exists() else {}
    adapter_issues=read(batch/'adapter-issue-audit.json') if (batch/'adapter-issue-audit.json').exists() else {}
    adapters_complete=True
    if require_complete and state['status']!='completed':raise ValueError('队列尚未结束')
    tables=[];details={};lines=['# 夜间算子体检与优化结果','','本批固定FlagGems/FlagTree，使用独立KGS Native合同的完整调用walltime计时。数值为性能用例相对等价PyTorch参考代码的几何平均；不与旧设备kernel成绩直接比较。','',
       '| 算子 | 原版来源 | 原版完整评测 | 相对PyTorch | 首个问题 |','|---|---|---:|---:|---|']
    records={r['operator']:r for r in plan['records']}
    for op,record in records.items():
        root=batch/'results'/op;leaf=batch/record['catalog_leaf']
        effective,leaf,path,retested=baseline_location(batch,record,adapter_issues.get(op))
        adapters_complete=adapters_complete and retested
        result=read(path) if path.exists() else {}
        label='FlagGems-sglang' if record['source']=='flaggems_sglang' else 'FlagGems'
        item={'operator':op,'upstream':label,'status':result.get('status','未完成'),
              'coverage':effective['coverage'],'baseline_source_sha256':effective['source_sha256'],
              'result_path':str(path),'adapter_issue':adapter_issues.get(op),'adapter_retest_available':retested,
              'passed_workloads':result.get('num_passed'),'workloads':result.get('num_workloads')}
        audit=reference_audits.get(op,{})
        item['reference_scope']=('cpu_fallback' if audit.get('cpu_fallback') else
                                 audit.get('reference_scope','native_npu'))
        if audit:item['reference_execution']=audit
        if result.get('status')=='PASSED':
            loaded=verify(path,effective,leaf)
            if loaded['source_sha256']!=effective['source_sha256']:raise ValueError('基线源码与plan不同')
            item['score']=statistics.geometric_mean(r['reference_latency_ms']/r['latency_ms'] for r in loaded['timing'].values())
            item['workloads']=len(loaded['rows'])
            item['correctness_cases']=loaded['correctness_count'];item['timing_cases']=len(loaded['timing'])
            if reference_audits.get(op,{}).get('cpu_fallback'):
                item['raw_score_against_cpu_fallback']=item.pop('score')
                item['reference_execution']=reference_audits[op]
        failed=next((row for row in result.get('per_workload',[]) if row.get('status')!='PASSED'),{})
        message=(failed.get('log') or result.get('log') or state.get('jobs',{}).get('baseline:'+op,{}).get('error',''))
        if reference_audits.get(op,{}).get('cpu_fallback'):
            message='PyTorch参考包含CPU回退；不列为全NPU对照成绩。'+message
        elif item['reference_scope']=='compound_pytorch_with_host_loop':
            message='组合PyTorch参考含Python循环和.item()同步；非单个融合kernel，不是新增优化收益。'+message
        if op in adapter_issues:
            message=('接口补测已完成；' if retested else '评测适配待补测；')+adapter_issues[op]['reason']+'；'+message
        item['first_error']=message
        short=(message.strip().splitlines()[-1] if message else '')[:100].replace('|','/').replace('\n',' ')
        if op in adapter_issues:
            short=(('适配补测完成：' if retested else '适配待补测：')+adapter_issues[op]['reason']+'；'+short)[:140].replace('|','/')
        if item['reference_scope']=='compound_pytorch_with_host_loop':
            short='组合PyTorch参考含expert循环与.item()同步；固定FlagGems成绩，非新增优化收益'
        coverage=('%s/%s'%(item['passed_workloads'],item['workloads'])) if item['workloads'] is not None else item['status']
        lines.append('| '+op+' | '+label+' | '+coverage+' | '+
          ('%.3f×'%item['score'] if 'score' in item else '—')+' | '+short+' |')
        tables.append(item)
    lines+=['','## 重点算子四组对照','',
        '各组收到相同固定原版源码作为起点上下文，KG优化预算每组2个正式轮次；原版已单独评测，首个生成轮次不自动等同原版。HG-KDA组是加载扩展分析角色的原生KG；独立规则控制器另行区分。']
    complete=adapters_complete
    for op in ['value_selecting_reduction_backward','moe_sum_reduce']:
        record=records[op];leaf=batch/record['catalog_leaf'];root=batch/'results'/op
        versions=[];basepath=root/'FlagGems/FlagGems-1.result.json'
        has_baseline=basepath.exists() and read(basepath).get('status')=='PASSED'
        if has_baseline:versions.append(('固定原版',verify(basepath,record,leaf)))
        else:complete=False
        groups={}
        for arm in ['kg-no-profile','kg-profile','HG-KDA']:
            paths=[root/arm/'independent'/(arm+'-%d.result.json'%i) for i in [1,2]]
            if not all(p.exists() and read(p).get('status')=='PASSED' for p in paths):
                groups[arm]={'status':'未完成或未通过'};complete=False;continue
            loaded=[verify(p,record,leaf) for p in paths]
            if loaded[0]['source_sha256']!=loaded[1]['source_sha256']:raise ValueError('最终两次复验源码不同')
            versions.extend([(arm+'-复验'+str(i),v) for i,v in enumerate(loaded,1)])
            groups[arm]={'status':'passed','source_sha256':loaded[0]['source_sha256'],'paths':[str(p.relative_to(batch)) for p in paths]}
            groups[arm].update(check_profile_evidence(root/arm))
            formal=groups[arm]['formal_rounds']
            groups[arm]['first_measured_source_matches_flaggems']=bool(
                formal and formal[0].get('source_sha256')==record['source_sha256'])
            if arm=='kg-profile':
                groups[arm]['profiling_analysis_verified']=bool(groups[arm]['verified_completed_analyses'])
                if not groups[arm]['profiling_analysis_verified']:complete=False
            semantics=root/arm/'supplementary-semantics.json'
            semantic_result=read(semantics) if semantics.exists() else {}
            groups[arm]['supplementary_semantics_result']={k:semantic_result.get(k) for k in
                ['status','passed','total','source_sha256','checker_sha256']}
            review=root/arm/'semantic-source-review.json'
            groups[arm]['source_semantic_review']=read(review) if review.exists() else None
            groups[arm]['supplementary_semantics_verified']=valid_semantics(
                semantic_result,op,loaded[0]['source_sha256'],
                hashlib.sha256(Path(__file__).with_name('check-nightly-semantics.py').read_bytes()).hexdigest())
            if not groups[arm]['supplementary_semantics_verified']:complete=False
            if arm=='HG-KDA':
                evidence=root/arm/'role-load-evidence.json'
                groups[arm]['role_load_evidence']=read(evidence) if evidence.exists() else None
                roles=(groups[arm]['role_load_evidence'] or {}).get('hgkda_roles_found',[])+groups[arm]['archived_hgkda_roles']
                for role in roles:
                    data=(root/arm/role['archived']).read_bytes()
                    if hashlib.sha256(data).hexdigest()!=role['sha256'] or 'HG-KDA 角色扩展' not in data.decode('utf-8'):
                        raise ValueError('实际加载的HG-KDA角色快照不匹配')
                # 角色存在仅证明加载；实际采集和结构化分析同样是闭环条件。
                groups[arm]['integration_verified']=bool(
                    roles and groups[arm]['verified_completed_analyses'] and
                    (groups[arm]['analyzer_invocations'] or groups[arm]['analyzer_delegations']))
                if not groups[arm]['integration_verified']:complete=False
        compared=comparison.compare(versions) if versions else []
        if not has_baseline:
            for row in compared:
                row['relative_master']=None
                row['reference_drift_geomean']=None
        details[op]={'groups':groups,'versions':compared}
        lines+=['','### '+op,'','| 版本 | 相对PyTorch | 相对固定原版 | PyTorch参考变化 |','|---|---:|---:|---:|']
        for row in compared:
            lines.append('| %s | %.3f× | %s | %s |'%(row['version'],row['relative_pytorch'],
                '%.3f×'%row['relative_master'] if row['relative_master'] is not None else '—',
                '%.3f×'%row['reference_drift_geomean'] if row['reference_drift_geomean'] is not None else '—'))
        for arm,value in groups.items():
            if value['status']!='passed':lines.append('| '+arm+' | 未完成或未通过 | — | — |')
        lines+=['','完整合同通过与补充语义通过分别列示；补测尚未通过的源码保留实验成绩，尚不视为最终有效交付。','',
                '| 模式 | 补充语义检查 | 已知源码语义问题 |','|---|---|---|']
        for arm,value in groups.items():
            if value['status']!='passed':continue
            sem=value['supplementary_semantics_result']
            label=('通过' if value['supplementary_semantics_verified'] else
                   '未通过' if sem.get('status')=='failed' else '尚未完成或身份未核实')
            if sem.get('total') is not None:label+=' %s/%s'%(sem.get('passed'),sem['total'])
            review=value['source_semantic_review'] or {}
            issue=''
            if review.get('source_sha256')==value['source_sha256'] and review.get('status')=='unsupported_valid_stride':
                issue='CPU包装层检查：合法转置stride被拒绝；NPU补测另列'
            lines.append('| '+arm+' | '+label+' | '+issue+' |')
        lines+=['','| 模式 | 正式轮次 | 诊断任务 | 完成的profiling分析 | 失败的采集请求 | 明确采纳profiling建议的轮次 | 首轮与FlagGems源码相同 |',
                '|---|---:|---:|---:|---:|---|---|']
        for arm,value in groups.items():
            if value['status']!='passed':continue
            guided='/'.join(str(r['round']) for r in value['profile_guided_rounds']) or '未登记'
            same='是' if value['first_measured_source_matches_flaggems'] else '否，生成候选另计'
            lines.append('| %s | %d | %d | %d | %d | %s | %s |'%(arm,len(value['formal_rounds']),len(value['debug_jobs']),len(value['verified_completed_analyses']),len(value['failed_profiles']),guided,same))
        if versions and has_baseline:
            args=[sys.executable,str(PROJECT/'tools/evaluation/compare-operator-versions.py')]
            for name,value in versions:args+=['--version',name,value['path']]
            args+=['--output',str(root/'版本对比.md')]
            subprocess.run(args,check=True,stdout=subprocess.PIPE)
    followups=[]
    if any(op in records for op in ['selu_','log10_']):
        lines+=['','## 追加的逐元素算子实验','','以下独立于两个重点算子的四组对照；原版精度失败时，不计算相对该失败原版的性能比。',
                '','| 算子 | 模式 | 两次完整复验 | 相对PyTorch | 状态 |','|---|---|---|---|---|']
    for op in ['selu_','log10_']:
        if op not in records:continue
        arm='HG-KDA';root=batch/'results'/op/arm
        paths=[root/'independent'/(arm+'-%d.result.json'%i) for i in [1,2]]
        job=state.get('jobs',{}).get('kg:'+op+':'+arm,{})
        item={'operator':op,'arm':arm,'status':job.get('status','pending'),'scores':[]}
        if all(p.exists() and read(p).get('status')=='PASSED' for p in paths):
            checked=[verify(p,records[op],batch/records[op]['catalog_leaf']) for p in paths]
            if checked[0]['source_sha256']!=checked[1]['source_sha256']:raise ValueError('追加算子复验源码不同')
            item.update(status='passed',source_sha256=checked[0]['source_sha256'],
                        workloads=len(checked[0]['rows']),result_sha256=[v['result_sha256'] for v in checked],
                        scores=[statistics.geometric_mean(r['reference_latency_ms']/r['latency_ms'] for r in v['timing'].values()) for v in checked])
            item['profile_evidence']=check_profile_evidence(root)
        elif job.get('status')=='passed':
            item.update(status='invalid',error='队列汇总与完整复验文件矛盾')
        else:item['error']=job.get('error','')
        followups.append(item)
        lines.append('| %s | %s | %s | %s | %s |'%(op,arm,
            ('%d/%d，两次均通过'%(item['workloads'],item['workloads'])) if item['status']=='passed' else '—',
            ' / '.join('%.3f×'%v for v in item['scores']) or '—',item['status']))
    result={'queue_status':state['status'],'all_priority_repeats_and_integration_verified':complete,
            'baseline_table':tables,'priority_comparisons':details,
            'followup_optimizations':followups,
            'note':'流水指标诊断与模拟器热点不等同定量Roofline；边界优化仍需一般语义检查和专家评审。'}
    (batch/'统计核验.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode())
    lines+=['','完整合同、源码与逐case结果随批次保存。未通过结果不计有效加速比；原版失败与本机接口/后端问题分别解释。']
    (batch/'结果汇总.md').write_bytes(('\n'.join(lines)+'\n').encode())
    # 严格门禁失败也保存当前诊断，避免调用者只看到旧报告和非零退出码。
    if require_complete and not complete:raise ValueError('体检适配补测、重点算子四组复验或HG-KDA实际集成证据不完整')
    return {'queue_status':state['status'],'correctness_passed':sum(x['status']=='PASSED' for x in tables),
            'native_reference_scores':sum('score' in x and x['reference_scope']=='native_npu' for x in tables),
            'compound_reference_scores':sum('score' in x and x['reference_scope']=='compound_pytorch_with_host_loop' for x in tables),
            'baseline_targets':len(tables),'all_priority_repeats_and_integration_verified':complete}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-dir',type=Path,required=True)
    p.add_argument('--require-complete',action='store_true')
    a=p.parse_args();print(json.dumps(build(a.batch_dir,a.require_complete),ensure_ascii=False))
