"""Build method-transfer context from current source, typed knowledge, contracts, and bound captures."""
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
    """Keep raw seconds and explicit microseconds per operator call to avoid unit/denominator confusion."""
    assert iterations>0
    return [dict(row,self_us_per_op=row['self_seconds']*1e6/iterations,
                 inclusive_us_per_op=row['cumulative_seconds']*1e6/iterations) for row in rows]


def compact_regions(data):
    """Keep region/interval statistics needed for decisions; read original PCs and idle intervals on demand."""
    if not data:return None
    timeline=dict(data['timeline'])
    if 'pipelines' in timeline:
        timeline['pipelines']=[dict(row,idle_intervals=row.get('idle_intervals',[])[:8]) for row in timeline['pipelines']]
    rows=[]
    for row in data['regions'][:8]:
        item={k:v for k,v in row.items() if k not in {'pcs','source_lines'}}
        item.update(pcs=row.get('pcs',[])[:8],pc_count=len(row.get('pcs',[])),
                    source_lines=row.get('source_lines',[])[:16])
        rows.append(item)
    return dict(source_sha256=data.get('source_sha256'),case=data.get('case'),regions=rows,
        total_regions=data['total_regions'],timeline=timeline,limitations=data['limitations'],
        compact_policy='At most 8 regions with 8 PCs each and 8 idle intervals per pipeline; see bound context for complete originals')


def retire_previous(workspace,digest):
    """Retain feedback from prior source without presenting it as current-source captures or scores."""
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
    # The model needs actionable methods and provenance summaries; the host retains complete historical paths and nested verification originals.
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
        assert item['source_sha256']==digest,'Paired capture does not belong to current source'
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
            scope='Keep the 32 most changed metrics and full transfer granularity; list actual IR separately without duplicating source and historical knowledge')
        # Detailed evidence from full context must reach model input, not merely be written to an unused file.
        deep=full['facts']
        for index,item in enumerate(deep.get('compiled_ir',[])):
            path=Path(item['path']);assert sha(path)==item['sha256']
            folder=workspace/'actual-ir'/digest[:12];folder.mkdir(parents=True,exist_ok=True)
            target=folder/('paired-'+str(index)+'-'+path.name);shutil.copyfile(path,target)
            if not any(row['sha256']==item['sha256'] for row in artifacts):
                artifacts.append(dict(path=str(target.relative_to(workspace)),sha256=item['sha256'],origin='current_paired_capture'))
        paired['instruction_hotspots']=deep.get('instruction_hotspots')
        region_data=deep.get('regions')
        paired['regions']=compact_regions(region_data)
        paired['roofline']=deep.get('roofline')
    methods=read(Path(config['method_manifest']))
    for item in methods['files']:
        assert sha(Path(config['package'])/item['path'])==item['sha256'],item['path']
    runtime=workspace/'runtime-api.json'
    runtime_text=None
    if runtime.exists():
        api=read(runtime);lines=['# Current runtime interface facts','',f'Original SHA256: {sha(runtime)}','']
        for item in api['sources']:
            lines.extend([f"## {item['module']}",f"Source: {item['path']}；SHA256：{item['sha256']}",'```python'])
            lines.extend(f"{row['line']}: {row['text']}" for row in item['excerpts'])
            lines.extend(['```',''])
        for item in api.get('launcher_classes',[]):lines.extend([f"## {item['name']}",'```python',item['source'],'```'])
        for name,item in api.get('runtime_hooks',{}).items():
            lines.extend([f'## {name}',f"Type: {item['type']}; active callbacks: {item.get('callback_count','unknown')}"])
            if item.get('class_source'):lines.extend(['```python',item['class_source'],'```'])
        runtime_text=workspace/'runtime-api-excerpts.md'
        runtime_text.write_text('\n'.join(lines)+'\n',encoding='utf8')
    launch=None
    probe=workspace/'launch-probe.json'
    if probe.exists():
        document=read(probe);assert document['source_sha256']==digest,'Launch diagnostics do not belong to current source'
        launch=dict(source_sha256=digest,artifact_sha256=sha(probe),
            calls=[dict(case=r['case'],iteration=r['iteration'],jit_calls=r['jit_calls'],
                        candidate_dict_sizes=r['state']['candidate_dict_sizes']) for r in document['records']],
            returned_compiled=document['returned_compiled'],runtime_hooks=document['runtime_hooks'],hook_records=document.get('hook_records',[]),
            jit_cache_structure=document['records'][-1]['state']['jit'],scope=document['scope'])
    host=None
    profile=workspace/'host-profile.json'
    if profile.exists():
        document=read(profile);assert document['source_sha256']==digest,'Host-stack sample does not belong to current source'
        host=dict(source_sha256=digest,artifact_sha256=sha(profile),path=profile.name,
            records=[dict(case=row['case'],iterations=row['iterations'],
                top_cumulative=profile_frames(row['top_cumulative'][:16],row['iterations']),
                top_self=profile_frames(row['top_self'][:16],row['iterations'])) for row in document['records']],
            units='self_us_per_op and inclusive_us_per_op are normalized by sampled iterations; raw seconds accumulate all iterations. Parent/child inclusive times cannot be added; sampling overhead prevents a formal latency breakdown.',
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
          'See starter-provenance.json for origin; distinguish improving an existing candidate from fresh upstream generation. CC/DeepSeek makes all candidate changes.',
          'Cumulative region instruction time is for ranking only; productive, wait_only, and idle are disjoint for the same core/window. Idle does not establish causal stalls.',
          'Do not choose edits from one slow instruction alone; combine surrounding loads, compute, stores, and actual IR into a measurable region-level mechanism hypothesis.',
          'The original FlagGems starter may have noncompliant device fallbacks; missing initial captures do not validate its Triton path.',
          'Use only the full KGS contract for formal performance; record profiling/device-task time separately from full-call latency.',
          'Keep actual IR, Roofline, and component metrics unknown when absent; do not invent measurements from source.',
          'All-zero pipeline/movement counters leave capture coverage unknown; they do not establish idle hardware or zero overhead.',
          'Do not divide whole-task work by per-core mean time; counts and durations must have matching aggregation scopes.',
          'Preflight success is a review handoff; continue candidate generation from feedback when formal performance is below target.'])
    (workspace/'context-brief.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return value
