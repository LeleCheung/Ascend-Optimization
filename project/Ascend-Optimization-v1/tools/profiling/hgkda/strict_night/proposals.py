"""把实际人工审查的Triton修正接入同一卡锁队列，不与CC工作区并发。"""
from pathlib import Path
from admission import static
from measurements import computation_identity,score


def process(runner):
    from night_common import read,save,sha,HERE,TOOLS,PY
    for spec_file in sorted((runner.batch/'proposals').glob('*/*/spec.json')):
        spec=read(spec_file);op=spec['operator'];out=spec_file.parent;source=out/'main.py'
        assert op in runner.q.records and out.parent.name==op
        key='proposal:'+op+':'+out.name
        if runner.state['jobs'].get(key,{}).get('status')=='completed':continue
        if spec.get('requires_qualified_parent'):
            parents=(runner.state['operators'].get(op) or {}).get('proposals',[])
            if not any(v.get('source_sha256')==spec['parent_source_sha256'] and v.get('qualification')=='passed' for v in parents):
                runner.state['jobs'][key]=dict(status='completed',result=dict(status='parent_not_qualified_no_measurement',
                    source=str(source),source_sha256=spec['source_sha256'],parent_source_sha256=spec['parent_source_sha256']))
                runner.persist();continue
        best=(runner.state['operators'].get(op) or {}).get('best') or {}
        if spec.get('defer_if_target_already_met') and best.get('qualification')=='passed' and best.get('search_score',0)>.8:
            runner.state['jobs'][key]=dict(status='completed',result=dict(status='retained_unmeasured_target_already_met',
                source=str(source),source_sha256=spec['source_sha256'],selected_source=best['source']))
            runner.persist();continue
        assert sha(source)==spec['source_sha256']
        manual=read(out/'review.json')
        assert manual['status']=='reviewed_strict_triton' and manual['source_sha256']==sha(source)
        assert manual['all_branches_reviewed'] and manual['dependencies_reviewed']
        if spec.get('host_glue'):
            from host_glue import admit
            save(out/'host-glue-admission.json',admit(source,out/'review.json'))
        else:assert not static(source.read_text()),'人工建议仍须通过严格Triton静态准入'
        # 只处理已有合格CANN参考的性能问题，不以较慢自写参考补齐名额。
        assert runner.q.records[op]['reference']=='qualified_CANN'
        normalized=computation_identity(source)
        assert normalized!=computation_identity(runner.batch/runner.q.records[op]['seed']),'排版变化不作为人工优化'
        duplicates=[]
        for workspace in (runner.q.runtime/'kg-runs').glob(op+'-*'):
            metadata=workspace/'.kernelgen/run-process.json'
            if metadata.exists():assert read(metadata).get('finished_at'),'不能在本算子KG运行中执行人工建议'
            for result in workspace.glob('stages/optimize/work/.kernelgen/evals/round-*/result.json'):
                candidate=result.with_name('main.py')
                if read(result).get('status')=='PASSED' and candidate.exists() and computation_identity(candidate)==normalized:
                    duplicates.append(str(result))
        if duplicates:
            runner.state['jobs'][key]=dict(status='completed',result=dict(status='duplicate_AST_not_remeasured',results=duplicates))
            runner.persist();continue
        runner.wait()
        previous_job=runner.state.get('current_job')
        runner.state['current_job']=key
        runner.state['jobs'][key]=dict(status='running',source=str(source),source_sha256=sha(source),attribution=spec['attribution'])
        runner.persist()
        runtime=out/'runtime-audit.json'
        if not runtime.exists():
            try:runner.cmd(key+'-runtime',[PY,TOOLS/'strict_triton/runtime_audit.py','--operator',op,
                '--source',source,'--leaf',runner.batch/'original-catalog/ops'/op,'--output',runtime],1200)
            except RuntimeError:
                if not runtime.exists():raise
        result=dict(status='runtime_rejected',runtime=str(runtime))
        if read(runtime)['status']=='passed':
            formal=out/'CANN/Codex-strict-CANN-1.result.json'
            if not formal.exists():
                runner.cmd(key+'-CANN',[PY,TOOLS/'cann_priority/evaluate.py','--operator',op,'--source',source,
                    '--leaf',runner.batch/'catalog/ops'/op,'--output',out/'CANN','--server',runner.q.server,
                    '--label','Codex-strict-CANN','--kind','strict_Triton_Codex_correction'])
            value=read(formal)
            result=dict(status=value.get('status'),result=str(formal))
            if value.get('status')=='PASSED' and not value.get('is_hack'):
                entry=dict(kind='Codex_proposal',source=str(source),source_sha256=sha(source),search_result=str(formal),
                    search_score=score(value),search_round_score=score(value),reference='qualified_CANN',
                    provenance=spec['attribution'],proposal_spec=str(spec_file),proposal_spec_sha256=sha(spec_file))
                if spec.get('host_glue'):entry['host_glue']=True
                entry.update(runner.validate(op,source,out))
                row=runner.state['operators'].setdefault(op,dict(attempts=[]))
                row.setdefault('proposals',[]).append(entry)
                result=dict(status='evaluated',entry=entry)
                if op=='fix':
                    leaf=runner.batch/'catalog/ops'/op
                    timing=[__import__('json').loads(x) for x in (leaf/'timing.jsonl').read_text().splitlines() if x.strip()]
                    # 明确采用原始大输入；父候选与修正版均按同一case诊断，不重复正式评测。
                    case=next(v for v in timing if v['name'].endswith('-3'))
                    parent=next((runner.batch/'admissions'/op).glob(spec['parent_source_sha256']+'/main.py'))
                    for label,path in [('parent',parent),('corrected',source)]:
                        probe=out/'components'/label;probe_spec=probe.with_name(label+'-spec.json')
                        payload=dict(operator=op,source=str(path),source_sha256=sha(path),leaf=str(leaf),case=case)
                        if probe_spec.exists():assert read(probe_spec)==payload
                        else:save(probe_spec,payload)
                        if not (probe/'result.json').exists():
                            assert not probe.exists(),'诊断目录未完成，保留检查'
                            runner.cmd(key+'-component-'+label,[PY,HERE/'component_probe.py','--spec',probe_spec,'--output',probe],1200)
        runner.state['jobs'][key].update(status='completed',result=result)
        runner.state['current_job']=previous_job
        runner.refresh_existing_scores();runner.verify_reviewed_best();runner.persist()
