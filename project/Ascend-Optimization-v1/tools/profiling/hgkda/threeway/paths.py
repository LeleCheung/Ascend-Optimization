"""对相同冻结输入采集实际设备路径；profiling耗时不替代KGS正式计时。"""
import hashlib
import json
import math
from pathlib import Path
from trace_summary import summarize


def process(r):
    from run import PY,quote,save_json
    from repair_references import process as repair
    repair(r)
    from cpp_references import process as cpp_references
    cpp_references(r)
    next_phase=Path(__file__).resolve().parent.parent/'next_phase'
    for op in ['log10_','hardswish_out','native_group_norm','cross_out','nan_to_num_']:
        leaf=r.batch/r.q.records[op]['catalog_leaf']
        cases=[json.loads(x) for x in (leaf/'timing.jsonl').read_text().splitlines() if x.strip()]
        selected=[x for x in cases if x.get('inputs',{}).get('spec',{}).get('dtype')=='float32']
        if not selected:selected=cases
        ordered=sorted(selected,key=lambda x:math.prod(x.get('inputs',{}).get('spec',{}).get('shape',[1])))
        picks=[ordered[0]]+([ordered[-1]] if len(ordered)>1 else [])
        if op in {'cross_out','nan_to_num_'}:picks=[ordered[-1]]
        sources=[('HG-KDA-Triton',r.starting_source(op),'candidate'),('PyTorch',r.starting_source(op),'pytorch')]
        best=r.state.get('operators',{}).get(op,{}).get('best') or {}
        if (best.get('status')=='PASSED' and best.get('supplementary_status')=='passed'
                and best.get('source_sha256')!=hashlib.sha256(r.starting_source(op).read_bytes()).hexdigest()):
            sources.append(('HG-KDA-best',Path(best['source']),'candidate'))
        reference=r.state.get('references',{}).get(op) or {}
        if reference.get('status')=='PASSED' and reference.get('supplementary_status')=='passed':
            sources.append(('CANN',Path(reference['source']),'candidate'))
        from comparison_sources import native_sources
        sources.extend((label, source, 'candidate') for label,source in native_sources(r.state,op))
        for index,case in enumerate(picks):
            for label,source,side in sources:
                sha=hashlib.sha256(source.read_bytes()).hexdigest()
                key='path:'+op+':'+str(index)+':'+label+':'+sha[:12]
                if r.state.get('jobs',{}).get(key,{}).get('status')=='completed':continue
                out=r.batch/'paths'/op/('case-'+str(index))/(label+'-'+sha[:12])
                spec=out/'spec.json';trace=out/'trace.json';result=out/'summary.json'
                out.mkdir(parents=True,exist_ok=True)
                payload=dict(operator=op,leaf=str(leaf),source=str(source),source_sha256=sha,case=case,
                             implementation_label=label,scope='原合同同输入/同seed的独立profiling；不冒充正式延迟')
                if spec.exists() and json.loads(spec.read_text(encoding='utf8'))!=payload:
                    raise r.module.Attention('路径采集输入身份已改变 '+str(spec))
                if not spec.exists():save_json(spec,payload)
                def capture(out=out,spec=spec,trace=trace,result=result,side=side,label=label,op=op,index=index):
                    r.q.run('path-'+op+'-'+str(index)+'-'+label,
                            quote([PY,next_phase/'driver.py','--spec',spec,'--side',side,'--trace',trace]),600)
                    value=summarize(trace);value['spec']=payload
                    save_json(result,value)
                    return {k:v for k,v in value.items() if k!='device_tasks'}
                r.job(key,capture)
        if op=='cross_out':
            for label,source,side in sources:
                if label not in {'HG-KDA-Triton','HG-KDA-best'}:continue
                sha=hashlib.sha256(source.read_bytes()).hexdigest()
                spec=r.batch/'paths'/op/'case-0'/(label+'-'+sha[:12])/'spec.json'
                out=r.batch/'actual-ir'/op/(label+'-'+sha[:12])
                def capture_ir(spec=spec,out=out,label=label,sha=sha):
                    r.q.run('actual-ir-cross-'+sha[:12],quote([PY,next_phase/'driver.py','--spec',spec,
                        '--side','candidate','--ir',out]),600)
                    manifest=out/'manifest.json';value=json.loads(manifest.read_text(encoding='utf8'))
                    if not value.get('entries'):raise r.module.Attention('实际编译IR为空 '+str(manifest))
                    return dict(manifest=str(manifest),entries=len(value['entries']))
                r.job('actual-ir:cross_out:'+sha[:12],capture_ir)
    here=Path(__file__).resolve().parent
    output=r.batch/'within-process/log10-fresh-input-native-v2.json'
    r.job('within-process:log10_:fresh-input-native-v2',lambda:r.q.run('within-process-log10-native-v2',
        quote([PY,here/'within_process.py','--batch',r.batch,'--output',output]),900))
    from reparse_paths import main as reparse
    reparse()
