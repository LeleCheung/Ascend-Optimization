"""当前证据的限额索引：给模型事实摘要与原件位置，不重复注入整份采集。"""
import hashlib
import json
from pathlib import Path


def select(row, keys):
    return {key:row[key] for key in keys if key in row}


def host_profile_context(directory, seed_sha256):
    """将来源核验的宿主诊断加入模型上下文，明确区分旧版本与当前种子。"""
    rows=[]
    for path in Path(directory).glob('host-profile-*.json'):
        raw=path.read_bytes();profile=json.loads(raw)
        source=Path(profile['source'])
        if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest()!=profile['source_sha256']:
            continue
        calls=profile.get('calls',0)
        if profile.get('status')!='completed' or not isinstance(calls,int) or calls<=0:continue
        rows.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
            source=profile['source'],source_sha256=profile['source_sha256'],matches_seed=profile['source_sha256']==seed_sha256,
            compiled_caches=profile.get('compiled_caches'),scope=profile.get('scope'),
            top_self_time=[dict(function=row['function'],calls=row['calls'],
                microseconds_per_profiled_call=row['self_seconds']*1.e6/calls) for row in profile.get('top_self_time',[])[:6]],
            interpretation='cProfile有额外开销；不代替KGS速度。matches_seed为false时仅是历史版本诊断，不能作为当前种子的已测数据'))
    rows.sort(key=lambda row:row['matches_seed'],reverse=True)
    return rows[:1]


def bounded_context(context, path, max_chars=32000):
    if max_chars<4000:raise ValueError('证据预算不得小于4000字符')
    raw=Path(path).read_bytes()
    result=select(context,['schema','status','spec','execution_guidance','limits','provenance'])
    result['full_evidence']=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
        characters=len(raw.decode('utf8')),usage='按需读取完整原件；摘要省略项不表示指标不存在')
    source=context.get('source',{})
    result['source']=select(source,['path','sha256','full_characters'])
    result['source'].update(text=source.get('text','')[:6000],read_full_on_demand=True)
    capture_quality={}
    capture_state=context.get('provenance',{}).get('capture_state')
    if capture_state and Path(capture_state).is_file():
        capture_raw=Path(capture_state).read_bytes()
        if hashlib.sha256(capture_raw).hexdigest()!=context['provenance']['sha256']:
            raise ValueError('采集状态与证据身份不同，不拼接新的采集状态')
        capture_quality=json.loads(capture_raw).get('sides',{})
    result['paths']={}
    for side,value in context.get('paths',{}).items():
        entry=select(value,['status','error','semantics','simulator_status','simulator_error','simulator_warnings'])
        entry.update(select(capture_quality.get(side,{}),['simulator_status','simulator_error','simulator_warnings']))
        entry['kernels']=[select(k,['name','basic']) for k in value.get('kernels',[])[:4]]
        entry['kernel_count']=len(value.get('kernels',[]))
        metrics=value.get('metrics',{})
        entry['metrics']=select(metrics,['path','sha256','scope'])
        entry['metrics']['kernels']=[dict(kernel=k['kernel'],
            selected_metrics=k.get('selected_metrics',[])[:9],granularity=k.get('granularity'),
            all_metric_count=k.get('all_metric_count')) for k in metrics.get('kernels',[])[:4]]
        result['paths'][side]=entry
    result['comparisons']=[]
    for comparison in context.get('comparisons',[])[:3]:
        entry=select(comparison,['reference_kind','candidate_kernel','reference_kernel','all_metric_count','selection','interpretation','quality_flag_count','full_comparison'])
        entry['selected_metrics']=[select(metric,['table','metric','aggregation','unit','reference','candidate',
            'absolute_delta','candidate_over_reference','status','reference_rows','candidate_rows','flags','family'])
            for metric in comparison.get('selected_metrics',[])[:8]]
        entry['selected_metric_count']=len(comparison.get('selected_metrics',[]))
        entry['derived']={name:select(row,['reference','candidate','absolute_delta','candidate_over_reference','unit'])
            for name,row in comparison.get('derived',{}).items() if row.get('candidate') is not None or row.get('reference') is not None}
        entry['quality_flags']=comparison.get('quality_flags',[])[:2]
        result['comparisons'].append(entry)
    ir=context.get('actual_ir',{})
    result['actual_ir']=dict(entry_count=len(ir.get('entries',[])),entries=[
        dict(symbol=entry.get('symbol'),artifacts=entry.get('artifacts',[])[:6]) for entry in ir.get('entries',[])[:4]])
    result['regions']=[]
    for region in context.get('regions',[])[:2]:
        entry=select(region,['schema','status','grouping','limitations','total_regions'])
        entry['regions']=[select(row,['kernel','core','kind','file','start','end','instructions','pipelines',
            'instruction_time_sum_us','largest_instruction_us','wait_instruction_time_sum_us','contains_function_entry',
            'trace_interval_union','trace_time_unit','interpretation']) for row in region.get('regions',[])[:3]]
        timeline=region.get('timeline',{})
        entry['timeline']=select(timeline,['status','reason','time_unit','capture_duration','capture_scope','interpretation'])
        entry['timeline']['pipelines']=[select(pipe,['pipeline','pipeline_kind','event_count','productive_ratio',
            'wait_only_ratio','idle_ratio','largest_idle']) for pipe in timeline.get('pipelines',[]) if pipe.get('pipeline_kind')=='hardware_execution']
        result['regions'].append(entry)
    result['budget']=dict(max_chars=max_chars,policy='指标保留绝对值和比率；原件带SHA；按需读完整源码/IR/区域')
    def size():return len(json.dumps(result,ensure_ascii=False,indent=2))
    if size()>max_chars:
        result['source'].pop('text',None)
        result['budget']['source_text_deferred']=True
    if size()>max_chars:
        for entry in result['paths'].values():
            entry['metrics'].pop('kernels',None)
            entry['kernels']=[dict(name=k.get('name')) for k in entry['kernels']]
        result['budget']['path_detail_deferred']=True
    if size()>max_chars:
        for comparison in result['comparisons']:
            comparison.pop('quality_flags',None)
        result['regions']=[]
        result['budget']['region_detail_deferred']=True
    if size()>max_chars:raise ValueError('证据摘要仍超预算；拒绝静默截断JSON')
    return result
