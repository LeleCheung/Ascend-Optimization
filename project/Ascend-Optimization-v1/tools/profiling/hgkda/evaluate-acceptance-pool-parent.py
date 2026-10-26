#!/usr/bin/env python3
"""同批原合同评测池化父版一次，按逐case实际延迟核验新分块收益。"""
from pathlib import Path
import hashlib
import json
import math
import os
import sys
sys.dont_write_bytecode=True


def main():
    import torch,torch_npu
    from kernelgen_client.catalog import Catalog
    from kernelgen_client.protocol.schema import Implementation,EvaluationSettings
    from kernelgen_server.evaluation.engine import EvaluationEngine
    from kernelgen_server.backends.ascend.device import NpuDevice
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7' and torch.npu.device_count()==1
    root=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1')
    batch=root/'operators/campaigns/acceptance-three-20261026'
    source=root/'operators/campaigns/remaining-five-20261023-v2/deliverables/reviewed/fractional_max_pool3d/main.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='4228f95c27e2dedb1edad3d8c37441a96e2c6384c84d423533be7413b30a0a66'
    output=batch/'validation-v2/fractional_max_pool3d/parent-comparison'
    output.mkdir(exist_ok=False)
    def save(name,data):(output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+chr(10))
    catalog=Catalog(batch/'catalog');definition=catalog.load('fractional_max_pool3d')
    impl=Implementation(name='acceptance-pool-parent',definition='fractional_max_pool3d',language='triton',entrypoint='main.py::run',sources=[dict(path='main.py',content=source.read_text())])
    request=catalog.request('fractional_max_pool3d',impl,EvaluationSettings(warmup_ms=1000,benchmark_ms=100,num_trials=3,timeout_seconds=1200))
    save('request.json',request.model_dump(mode='json'))
    engine=EvaluationEngine(NpuDevice(perf_mode='walltime',num_trials=3),'npu:0','ascend',oracle_path=definition.oracle_path)
    preflight=engine.preflight(request);save('preflight.json',preflight);assert preflight['status']=='PASSED'
    result=engine.evaluate(request).model_dump(mode='json');save('result.json',result)
    assert result['status']=='PASSED' and result['num_passed']==result['num_workloads']==9 and not result['is_hack']
    final=json.loads((output.parent/'default-reference.result.json').read_text())
    a={r['uuid']:r['latency_ms'] for r in result['per_workload'] if r['phase']=='timing'}
    b={r['uuid']:r['latency_ms'] for r in final['per_workload'] if r['phase']=='timing'}
    assert set(a)==set(b) and len(a)==6
    score=math.exp(sum(math.log(a[k]/b[k]) for k in a)/len(a))
    summary=dict(status='passed',scope='同批同case完整调用实际延迟比，不用相对PyTorch汇总分相除',
        actual_speedup=score,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        final_source_sha256=hashlib.sha256((batch/'deliverables/fractional_max_pool3d/main.py').read_bytes()).hexdigest(),
        parent_score=result['geo_mean'],per_case=[dict(case=k,parent_latency_ms=a[k],final_latency_ms=b[k],actual_speedup=a[k]/b[k]) for k in a])
    save('comparison.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
