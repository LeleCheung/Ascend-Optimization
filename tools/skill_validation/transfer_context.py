"""为多算子方法迁移整理当前源码、分类知识、合同与已绑定的实际采集。"""
import hashlib
import json
import math
from pathlib import Path
import shutil

from hgkda.artifacts import unwrap
from hgkda.context import knowledge


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profile_frames(rows,iterations):
    """保留原始秒数，并显式给出每次算子调用的微秒数，避免单位与分母混淆。"""
    assert iterations>0
    return [dict(row,self_us_per_op=row['self_seconds']*1e6/iterations,
                 inclusive_us_per_op=row['cumulative_seconds']*1e6/iterations) for row in rows]


def retire_previous(workspace,digest):
    """保留上一源码的反馈原件，避免把它冒充当前源码的采集或成绩。"""
    for name in ['paired-context-pointer.json','performance-feedback.json','next-performance-feedback.json','latest-preflight.json','launch-probe.json','host-profile.json','compiler-comparison.json']:
        path=workspace/name
        if not path.exists():continue
        value=unwrap(read(path));declared=value.get('source_sha256')
        if not declared or declared==digest:continue
        folder=workspace/'retired-evidence';folder.mkdir(exist_ok=True)
        target=folder/(name.removesuffix('.json')+'-'+sha(path)[:16]+'.json')
        if target.exists():assert target.read_bytes()==path.read_bytes()
        else:shutil.copyfile(path,target)
        path.unlink()


def build(config):
    run=Path(config['run_directory']);workspace=run/'workspace'
    source=workspace/config.get('active_source','starter.py');digest=sha(source)
    operator=config['job']['operator'];library=knowledge(operator,hypotheses=True)
    # 模型只需可执行方法与来源摘要；完整历史路径和嵌套验证原件由宿主保管。
    visible=[{k:record[k] for k in ['id','operators','types','status','signal','mechanism','action','constraints',
              'match','application','source_sha256'] if k in record} for record in library['mechanisms']]
    for name in ['knowledge.json','contract.json','reference-qualification.json']:
        target=workspace/name
        if name=='knowledge.json':
            target.write_text(json.dumps(dict(operator_type=library['operator_type'],mechanisms=visible,
                retrieval=library['retrieval'],update_policy=library['update_policy']),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    compiled=run/'attempts'/digest/'compiled/verification.json'
    artifacts=[]
    if compiled.exists():
        record=read(compiled);assert record['source_sha256']==digest
        target_root=workspace/'actual-ir'/digest[:12];target_root.mkdir(parents=True,exist_ok=True)
        for index,entry in enumerate(record['entries']):
            for item in entry['artifacts']:
                if item['kind'] not in ['ttir','bcmlir']:continue
                path=Path(item['path']);assert sha(path)==item['sha256']
                target=target_root/(str(index)+'-'+path.name);shutil.copyfile(path,target)
                artifacts.append(dict(path=str(target.relative_to(workspace)),sha256=item['sha256'],kernel=entry['symbol']))
    preflight=None
    if (workspace/'latest-preflight.json').exists():
        row=read(workspace/'latest-preflight.json')
        if row.get('source_sha256')==digest:preflight=row
    feedback=None
    for name in ['next-performance-feedback.json','performance-feedback.json']:
        if (workspace/name).exists():
            feedback=unwrap(read(workspace/name));assert feedback['source_sha256']==digest
            break
    paired=None
    pointer=workspace/'paired-context-pointer.json'
    if pointer.exists():
        item=read(pointer)
        assert item['source_sha256']==digest,'成对采集不属于当前源码'
        context=Path(item['context']);assert sha(context)==item['sha256']
        full=unwrap(read(context));facts=full['facts']['paired']
        selected=[row for row in facts['differences'] if any(word in row['metric'] for word in
                  ['ratio','time','instructions','datas','conflict','wait'])]
        def deviation(row):
            ratio=row.get('candidate_over_reference')
            return abs(math.log(ratio)) if isinstance(ratio,(int,float)) and ratio>0 else 0
        selected.sort(key=deviation,reverse=True)
        paired=dict(case=full['case'],source_sha256=full['source']['sha256'],
            reference_kind=full['reference_kind'],differences=selected[:32],
            task_information={side:facts[side]['basic'] for side in ['candidate','reference']},
            transfer_granularity=facts['derived_differences'],execution_groups=facts['execution_groups'],
            hypotheses=full['hypotheses'],limitations=facts['limitations'],
            full_context_sha256=item['sha256'],
            scope='保留变化最大的32项指标及完整搬运粒度；实际IR另列，不重复嵌入源码和历史知识')
    methods=read(Path(config['method_manifest']))
    for item in methods['files']:
        assert sha(Path(config['package'])/item['path'])==item['sha256'],item['path']
    runtime=workspace/'runtime-api.json'
    runtime_text=None
    if runtime.exists():
        api=read(runtime);lines=['# 当前运行时接口事实','',f'原件SHA256：{sha(runtime)}','']
        for item in api['sources']:
            lines.extend([f"## {item['module']}",f"来源：{item['path']}；SHA256：{item['sha256']}",'```python'])
            lines.extend(f"{row['line']}: {row['text']}" for row in item['excerpts'])
            lines.extend(['```',''])
        for item in api.get('launcher_classes',[]):lines.extend([f"## {item['name']}",'```python',item['source'],'```'])
        for name,item in api.get('runtime_hooks',{}).items():
            lines.extend([f'## {name}',f"类型：{item['type']}；活动回调数：{item.get('callback_count','unknown')}"])
            if item.get('class_source'):lines.extend(['```python',item['class_source'],'```'])
        runtime_text=workspace/'runtime-api-excerpts.md'
        runtime_text.write_text('\n'.join(lines)+'\n',encoding='utf8')
    launch=None
    probe=workspace/'launch-probe.json'
    if probe.exists():
        document=read(probe);assert document['source_sha256']==digest,'启动诊断不属于当前源码'
        launch=dict(source_sha256=digest,artifact_sha256=sha(probe),
            calls=[dict(case=r['case'],iteration=r['iteration'],jit_calls=r['jit_calls'],
                        candidate_dict_sizes=r['state']['candidate_dict_sizes']) for r in document['records']],
            returned_compiled=document['returned_compiled'],runtime_hooks=document['runtime_hooks'],hook_records=document.get('hook_records',[]),
            jit_cache_structure=document['records'][-1]['state']['jit'],scope=document['scope'])
    host=None
    profile=workspace/'host-profile.json'
    if profile.exists():
        document=read(profile);assert document['source_sha256']==digest,'宿主栈采样不属于当前源码'
        host=dict(source_sha256=digest,artifact_sha256=sha(profile),path=profile.name,
            records=[dict(case=row['case'],iterations=row['iterations'],
                top_cumulative=profile_frames(row['top_cumulative'][:16],row['iterations']),
                top_self=profile_frames(row['top_self'][:16],row['iterations'])) for row in document['records']],
            units='self_us_per_op与inclusive_us_per_op按采样迭代数归一化；原始seconds为全部迭代累计，父子inclusive不可相加；有采样开销，不等于正式耗时分解',
            scope=document['scope'])
    compiler=None
    path=workspace/'compiler-comparison.json'
    if path.exists():
        document=read(path);assert document['source_sha256']==digest
        compiler=dict(path=path.name,sha256=sha(path),source_sha256=digest,
            left_source_sha256=document['left_source_sha256'],all_normalized_ir_equal=document['all_normalized_ir_equal'],
            comparisons=len(document['comparisons']),changed=[row for row in document['comparisons'] if not row['equal']],scope=document['scope'])
    value=dict(operator=operator,phase=config['phase'],source_sha256=digest,active_source=source.name,
        contract=read(workspace/'contract.json'),knowledge_retrieval=library['retrieval'],
        mechanisms=visible,formal_feedback=feedback,preflight_feedback=preflight,actual_ir=artifacts,
        paired_evidence=paired,launch_diagnosis=launch,host_diagnosis=host,compiler_comparison=compiler,
        runtime_api=dict(path=runtime_text.name,sha256=sha(runtime_text),raw_json=runtime.name,raw_sha256=sha(runtime)) if runtime_text else None,
        method_manifest_sha256=sha(Path(config['method_manifest'])),
        interpretation_rules=[
          '知识是方法及条件，不提供历史最佳源码；候选全部修改由CC／DeepSeek完成。',
          '原始FlagGems起点可能含目标设备不合规回退，初次采集缺失不代表已验证其Triton路径。',
          '正式性能只采用KGS完整合同；profiling或设备任务时间与完整调用分别记录。',
          '缺少实际IR、Roofline或组件指标时保留未知，不根据源码虚构测量。',
          '采集流水和搬运计数全为零时保留采集覆盖未知，不能据此断言硬件空闲或没有开销。',
          '不得把全任务工作量除以逐核平均时间；计数与时间的聚合范围必须一致。',
          '预检通过是交接审查的边界，正式性能未达标须依据反馈继续生成候选。'])
    (workspace/'context-brief.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return value
