"""同合同CANN对照；结果只入参考列，不参与HG-KDA候选择优。"""
import hashlib
import json
from pathlib import Path


def process(r):
    from run import PY,quote,save_json
    here=Path(__file__).resolve().parent
    binding=(here/'cann_async.py').read_text(encoding='utf8')
    signatures={
        'log10_':"def run(A):\n    return _NATIVE.run(A=A)\n",
        'hardswish_out':"def run(x, *, out):\n    return _NATIVE.run(x=x,out=out)\n",
        'native_group_norm':"def run(input,weight,bias,N,C,HxW,group,eps):\n    return _NATIVE.run(input=input,weight=weight,bias=bias,N=N,C=C,HxW=HxW,group=group,eps=eps)\n",
    }
    for op,signature in signatures.items():
        out=r.batch/'references'/op
        source=out/'main.py'
        content=binding+'\n_NATIVE=Native('+repr(op)+')\n'+signature
        out.mkdir(parents=True,exist_ok=True)
        if source.exists() and source.read_text(encoding='utf8')!=content:
            raise r.module.Attention('参考包装已冻结；修改必须另建参考版本 '+op)
        if not source.exists():source.write_text(content,encoding='utf8')
        provenance=out/'provenance.json'
        if not provenance.exists():
            code='import importlib.util,json;from pathlib import Path;s=importlib.util.spec_from_file_location("cann_wrapper",'+repr(str(source))+');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);Path('+repr(str(provenance))+').write_text(json.dumps(m._NATIVE.provenance(),ensure_ascii=False,indent=2),encoding="utf8");p=Path(m.ROOT/"include"/m._NATIVE.record["header"]);Path('+repr(str(out/'header.h'))+').write_bytes(p.read_bytes())'
            r.job('reference-provenance:'+op,lambda op=op,code=code:r.q.run('reference-provenance-'+op,quote([PY,'-c',code]),180))
        def evaluate(op=op,source=source,out=out):
            r.q.run('reference-evaluate-'+op,quote([PY,here/'evaluate_reference.py',
                '--operator',op,'--source',source,'--leaf',r.batch/r.q.records[op]['catalog_leaf'],
                '--output',out/'formal','--server',r.q.server]),2100)
            path=out/'formal/CANN-async-python-1.result.json'
            value=json.loads(path.read_text(encoding='utf8'))
            return dict(status=value.get('status'),geo_mean=value.get('geo_mean'),num_passed=value.get('num_passed'),
                        num_workloads=value.get('num_workloads'),result=str(path),source=str(source),
                        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        is_hack=value.get('is_hack'),hack_reason=value.get('hack_reason'))
        result=r.job('reference-evaluate:'+op,evaluate)
        if result and result.get('status')=='PASSED':
            gates=out/'extra-checks.json'
            qualification=r.job('reference-extra:'+op,lambda op=op,source=source,gates=gates:r.supplementary(op,source,gates,'reference-extra-'+op))
            result.update(extra_checks=str(gates),supplementary_status=qualification,
                          provenance=str(provenance),reference_kind='CANN_native_aclnn',
                          scope='相同KGS完整调用；包含本Python ctypes事件管理成本，不称最佳宿主实现')
        r.state.setdefault('references',{})[op]=result
        r.save()
    # 源码可核验的自写Ascend C候选，单独记实现后端，仍沿用同一KGS合同。
    native=r.batch/'native-candidates/log10_/ascendc-ln-v1'
    manifest=native/'manifest.json'
    if manifest.exists():
        identity=json.loads(manifest.read_text(encoding='utf8'))
        for name,digest in identity['files'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
                raise r.module.Attention('Ascend C候选附件身份改变 '+name)
        source=native/'main.py'
        gates=native/'extra-checks.json'
        qualification=r.job('native-candidate-extra:log10_:ascendc-ln-v1',
            lambda:r.supplementary('log10_',source,gates,'native-log10-extra'))
        if qualification=='passed':
            def evaluate_native():
                r.q.run('native-evaluate-log10',quote([PY,here/'evaluate_reference.py',
                    '--operator','log10_','--source',source,'--leaf',r.batch/r.q.records['log10_']['catalog_leaf'],
                    '--output',native/'formal','--server',r.q.server,'--label','HG-KDA-AscendC',
                    '--kind',identity['backend']]),2100)
                path=native/'formal/HG-KDA-AscendC-1.result.json'
                value=json.loads(path.read_text(encoding='utf8'))
                return dict(status=value.get('status'),geo_mean=value.get('geo_mean'),num_passed=value.get('num_passed'),
                    num_workloads=value.get('num_workloads'),result=str(path),source=str(source),
                    source_sha256=identity['source_sha256'],extra_checks=str(gates),supplementary_status=qualification,
                    backend=identity['backend'],native_manifest=str(manifest),attribution=identity['attribution'],
                    latency_ms={x['uuid']:x['latency_ms'] for x in value.get('per_workload',[]) if x.get('phase')=='timing' and x.get('latency_ms') is not None})
            result=r.job('native-candidate-evaluate:log10_:ascendc-ln-v1',evaluate_native)
            r.state.setdefault('native_candidates',{})['log10_']=result
            row=r.state['operators'].setdefault('log10_',{})
            r.consider_best(row,result)
            if row.get('best',{}).get('source_sha256')==identity['source_sha256']:
                row['status']='Ascend C混合实现达标' if result.get('geo_mean',0)>.8 else 'Ascend C混合实现正确，性能待优化'
            r.save()
    from paths import process as capture_paths
    capture_paths(r)
