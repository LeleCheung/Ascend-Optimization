#!/usr/bin/env python3
"""宿主持卡锁，独立复现三项收尾中的池化或共轭，明确选择原参考或NPU分解。"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path('/data/hanle/ascend-optimization')
PROJECT=ROOT/'Ascend-Optimization/project/Ascend-Optimization-v1'
BATCH=PROJECT/'operators/campaigns/acceptance-three-20261026'


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+chr(10))


EVALUATE=r'''
from pathlib import Path
import sys,os,json,hashlib
sys.dont_write_bytecode=True
import torch,torch_npu
from kernelgen_client.catalog import Catalog
from kernelgen_client.protocol.schema import Implementation,EvaluationSettings
from kernelgen_server.evaluation.engine import EvaluationEngine
from kernelgen_server.backends.ascend.device import NpuDevice
assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7' and torch.npu.device_count()==1
assert not torch.are_deterministic_algorithms_enabled()
batch=Path(parameters['batch']);source=batch/'deliverables'/parameters['operator']/'main.py';out=Path(parameters['output'])
assert hashlib.sha256(source.read_bytes()).hexdigest()==parameters['source_sha256']
root=batch/('catalog' if parameters['reference']=='default' else 'npu-decomposition-catalog-v2')
catalog=Catalog(root);definition=catalog.load(parameters['operator'])
implementation=Implementation(name='acceptance-reproduction',definition=parameters['operator'],language='triton',entrypoint='main.py::run',sources=[dict(path='main.py',content=source.read_text())])
request=catalog.request(parameters['operator'],implementation,EvaluationSettings(warmup_ms=1000,benchmark_ms=100,num_trials=3,timeout_seconds=1200))
def save(name,data):(out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+chr(10))
save('conditions.json',dict(parameters,physical_card=7,timing='KGS完整调用walltime',catalog_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'ops'/parameters['operator']).iterdir() if p.is_file()}))
save('evaluation-request.json',request.model_dump(mode='json'))
engine=EvaluationEngine(NpuDevice(perf_mode='walltime',num_trials=3),'npu:0','ascend',oracle_path=definition.oracle_path)
preflight=engine.preflight(request);save('preflight.json',preflight);assert preflight['status']=='PASSED'
result=engine.evaluate(request).model_dump(mode='json');save('result.json',result)
print(json.dumps({k:result.get(k) for k in ['status','geo_mean','num_passed','num_workloads']}),flush=True)
assert result['status']=='PASSED' and not result['is_hack'] and result['num_passed']==result['num_workloads']
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operator',required=True,choices=['fractional_max_pool3d','_conj_physical'])
    p.add_argument('--reference',required=True,choices=['default','npu-decomposition'])
    p.add_argument('--label',required=True)
    a=p.parse_args()
    assert a.label.replace('-','').replace('_','').isalnum()
    manifest=json.loads((BATCH/'final-delivery.json').read_text())
    row=manifest['operators'][a.operator]
    source=BATCH/row['source']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==row['source_sha256']
    out=BATCH/'reproductions'/a.label
    assert not out.exists(),'复现输出不能覆盖'
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def idle():
            for port in [19655,19657,19659]:
                state=json.load(opener.open('http://127.0.0.1:%d/status'%port,timeout=15))['scheduler']
                assert state['healthy']==1 and not any(state[k] for k in ['active','waiting','broken'])
        idle()
        out.mkdir(parents=True)
        parameters=dict(batch=str(BATCH),output=str(out),operator=a.operator,reference=a.reference,source_sha256=row['source_sha256'])
        script='parameters='+repr(parameters)+chr(10)+EVALUATE
        request=dict(command=['{python}','run.py'],files=[dict(path='run.py',content=script)],timeout_seconds=1500)
        save(out/'debug-request.json',request)
        response=json.load(opener.open(urllib.request.Request('http://127.0.0.1:19659/debug/jobs',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'},method='POST'),timeout=1545))
        save(out/'debug-response.json',response)
        idle()
        print(json.dumps(dict(output=str(out),status=response['status'],job_id=response['job_id'])))
        assert response['status']=='SUCCEEDED'


if __name__=='__main__':main()
