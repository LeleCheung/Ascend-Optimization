"""复用已有完整测量；同SHA不同有效测量取几何平均，缓存副本不重复计数。"""
import hashlib
import json
import math
from pathlib import Path


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(result):
    assert result.get('status')=='PASSED' and not result.get('is_hack')
    assert result.get('num_workloads') and result['num_passed']==result['num_workloads']
    rows=result.get('per_workload',[])
    assert len(rows)==result['num_workloads'] and all(v['status']=='PASSED' for v in rows)
    timing=[v for v in rows if v.get('phase')=='timing']
    assert timing and all(v.get('latency_ms',0)>0 and v.get('reference_latency_ms',0)>0 for v in timing)
    value=math.exp(sum(math.log(v['reference_latency_ms']/v['latency_ms']) for v in timing)/len(timing))
    assert math.isclose(value,result['geo_mean'],rel_tol=1e-8,abs_tol=1e-10),'原始逐case与geo_mean不符'
    return value


def ledger(workspace,source):
    work=Path(workspace)/'stages/optimize/work';digest=sha(source)
    entries=[];failures=[];fingerprints=set();names=None
    for file in sorted(work.glob('.kernelgen/evals/round-*/result.json')):
        candidate=file.with_name('main.py')
        if not candidate.exists() or sha(candidate)!=digest:continue
        result=read(file)
        identity=read(file.with_name('identity.json'))
        fingerprints.add(identity['evaluation_fingerprint'])
        entries.append((file,result,'search'))
    for file in sorted(work.glob('.kernelgen/retests/'+digest+'/*/attempt-*.json')):
        value=read(file)
        assert value['solution_sha256']==digest
        if value['evaluation_fingerprint'] not in fingerprints:continue
        settings=value['contract']['settings']
        assert all(settings[k]==v for k,v in {'warmup_ms':1000,'benchmark_ms':100,'num_trials':3}.items())
        assert value['service']['timing']=='walltime'
        entries.append((file,value.get('result') or {},'KG_existing_retest'))
    observations=[];seen={};duplicates=[]
    for file,result,kind in entries:
        try:
            current=score(result)
            cases={(v['uuid'],v.get('phase')) for v in result['per_workload']}
            if names is None:names=cases
            assert names==cases,'同源码评测合同case身份不同'
        except (AssertionError,KeyError,TypeError,ValueError) as error:
            failures.append(dict(path=str(file),kind=kind,status=result.get('status'),error=str(error)));continue
        # 原生final retest若命中本批内容缓存，逐case原始数值相同，不伪称第二次实测。
        identity=hashlib.sha256(json.dumps(result['per_workload'],sort_keys=True).encode()).hexdigest()
        if identity in seen:
            duplicates.append(dict(path=str(file),same_measurement_as=seen[identity]));continue
        seen[identity]=str(file)
        observations.append(dict(path=str(file),sha256=sha(file),kind=kind,score=current))
    assert observations,'没有同源码的完整有效实测'
    return dict(status='passed' if not failures else 'formal_failure',source_sha256=digest,
                score=math.exp(sum(math.log(v['score']) for v in observations)/len(observations)),
                observations=observations,duplicates=duplicates,failures=failures,
                policy='只复用已存在的同源码/合同/设置测量；有效结果几何平均，缓存副本去重；不重新测量挑分')
