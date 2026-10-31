"""对绑定源码的代表输入保存设备路径与实际Triton IR；正式加速比另列。"""
import hashlib
import json
import math
from pathlib import Path


def collect(r):
    from run import PY,HERE,quote,save_json
    import sys
    sys.path.insert(0,str(HERE.parent/'threeway'))
    from trace_summary import summarize
    for op in r.q.records:
        leaf=r.batch/r.q.records[op]['catalog_leaf']
        cases=[json.loads(x) for x in (leaf/'timing.jsonl').read_text().splitlines() if x.strip()]
        cases=[x for x in cases if x['inputs']['spec']['dtype']=='float32'] or cases
        cases=sorted(cases,key=lambda x:math.prod(x['inputs']['spec']['shape']))
        picks=[cases[0]]+([cases[-1]] if len(cases)>1 else [])
        starting=r.starting_source(op)
        best=r.state.get('operators',{}).get(op,{}).get('best') or {}
        sources=[('starting-Triton',starting,'candidate'),('PyTorch',starting,'pytorch')]
        if best.get('source_sha256') and best['source_sha256']!=hashlib.sha256(starting.read_bytes()).hexdigest():
            sources.append(('HG-KDA',Path(best['source']),'candidate'))
        for index,case in enumerate(picks):
            for label,source,side in sources:
                sha=hashlib.sha256(source.read_bytes()).hexdigest()
                out=r.batch/'paths'/op/('case-'+str(index))/(label+'-'+sha[:12]);out.mkdir(parents=True,exist_ok=True)
                spec=out/'spec.json';trace=out/'trace.json';summary=out/'summary.json'
                payload=dict(operator=op,leaf=str(leaf),source=str(source),source_sha256=sha,case=case,
                    implementation_label=label,scope='原合同同输入/seed独立profiling；不替代正式KGS延迟')
                if spec.exists() and json.loads(spec.read_text())!=payload:raise r.module.Attention('路径输入不同')
                if not spec.exists():save_json(spec,payload)
                def capture(op=op,index=index,label=label,spec=spec,trace=trace,summary=summary,side=side):
                    if not trace.exists():r.q.run('path-'+op+'-'+str(index)+'-'+label,
                        quote([PY,HERE/'driver.py','--spec',spec,'--side',side,'--trace',trace]),600)
                    value=summarize(trace);save_json(summary,value)
                    return dict(summary=str(summary),tasks=len(value.get('device_tasks',[])))
                r.job('path:'+op+':'+str(index)+':'+label+':'+sha[:12],capture)
                if side=='candidate' and label=='starting-Triton' and index==0:
                    ir=r.batch/'actual-ir'/op/sha[:12]
                    def capture_ir(op=op,spec=spec,ir=ir):
                        manifest=ir/'manifest.json'
                        if not manifest.exists():r.q.run('actual-ir-'+op,quote([PY,HERE/'driver.py','--spec',spec,'--side','candidate','--ir',ir]),600)
                        value=json.loads(manifest.read_text())
                        if not value.get('entries'):raise RuntimeError('实际IR为空 '+op)
                        return dict(manifest=str(manifest),entries=len(value['entries']))
                    r.job('actual-ir:'+op+':'+sha[:12],capture_ir)
