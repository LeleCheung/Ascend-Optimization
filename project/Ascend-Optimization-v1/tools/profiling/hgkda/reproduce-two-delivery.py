#!/usr/bin/env python3
"""在910B宿主持卡锁，使用原KGS引擎独立复现两算子最终交付；保留新输出。"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import urllib.request

ROOT = Path('/data/hanle/ascend-optimization')
TOOLS = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator', choices=['upsample_bilinear2d', 'cumulative_trapezoid'], required=True)
    parser.add_argument('--label', required=True, help='新的输出名称，只含字母、数字、横线或下划线')
    parser.add_argument('--deterministic', action='store_true', help='显式开启官方确定性条件，仅累计积分使用')
    args = parser.parse_args()
    assert args.label.replace('-', '').replace('_', '').isalnum()
    assert args.deterministic == (args.operator == 'cumulative_trapezoid'), '累计积分必须显式选择确定性条件；上采样使用默认条件'
    batch = ROOT / 'Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024'
    manifest = json.loads((batch / 'final-delivery.json').read_text())
    selected = manifest['selected'][args.operator]
    source = batch / selected['delivered']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == selected['source_sha256']
    output = batch / 'reproductions' / args.label
    assert not output.exists(), '复现输出不能覆盖'
    with (ROOT / 'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = importlib.util.spec_from_file_location('reproduction_queue', TOOLS / 'run-nightly.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        q = module.Queue(batch.name, ROOT, 'tle_yy')
        q.idle()
        output.mkdir(parents=True)
        parameters = dict(batch=str(batch), source=str(source), output=str(output),
                          operator=args.operator, deterministic=args.deterministic, tools=str(TOOLS))
        script = 'parameters=' + repr(parameters) + '\n' + EVALUATE
        request = dict(command=['{python}', 'reproduce.py'],
                       files=[dict(path='reproduce.py', content=script)], timeout_seconds=1800)
        (output / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2) + '\n')
        response = json.load(q.opener.open(urllib.request.Request(q.server + '/debug/jobs',
            data=json.dumps(request).encode(), headers={'Content-Type': 'application/json'}, method='POST'), timeout=1845))
        (output / 'response.json').write_text(json.dumps(response, ensure_ascii=False, indent=2) + '\n')
        q.idle()
        print(json.dumps(dict(output=str(output), status=response['status'], job_id=response['job_id'])))
        if response['status'] != 'SUCCEEDED':
            raise SystemExit(1)


EVALUATE = r'''
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,importlib.util,json,os
import torch,torch_npu
from kernelgen_client.catalog import Catalog
from kernelgen_client.protocol.schema import Implementation,EvaluationSettings
from kernelgen_server.evaluation.engine import EvaluationEngine
from kernelgen_server.backends.ascend.device import NpuDevice
assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7' and torch.npu.device_count()==1
batch=Path(parameters['batch']);source=Path(parameters['source']);out=Path(parameters['output']);op=parameters['operator']
torch.use_deterministic_algorithms(parameters['deterministic'])
catalog=Catalog(batch/'catalog');definition=catalog.load(op)
impl=Implementation(name='two-source-reproduction',definition=op,language='triton',entrypoint='main.py::run',sources=[dict(path='main.py',content=source.read_text())])
request=catalog.request(op,impl,EvaluationSettings(warmup_ms=1000,benchmark_ms=100,num_trials=3,timeout_seconds=1800))
conditions=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),deterministic=parameters['deterministic'],physical_device=7,timing_scope='KGS原Native引擎完整调用walltime',contract_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (batch/'catalog/ops'/op).iterdir() if p.is_file()})
(out/'conditions.json').write_text(json.dumps(conditions,ensure_ascii=False,indent=2)+'\n')
(out/'evaluation-request.json').write_text(json.dumps(request.model_dump(mode='json'),ensure_ascii=False,indent=2)+'\n')
engine=EvaluationEngine(NpuDevice(perf_mode='walltime',num_trials=3),'npu:0','ascend',oracle_path=definition.oracle_path)
preflight=engine.preflight(request)
(out/'preflight.json').write_text(json.dumps(preflight,ensure_ascii=False,indent=2)+'\n')
if preflight['status']!='PASSED':raise SystemExit('准入未通过')
checks=['check-remaining-five.py','check-remaining-special.py'] if op=='upsample_bilinear2d' else ['check-endurance.py']
for name in checks:
    spec=importlib.util.spec_from_file_location('reproduction_gate_'+name,Path(parameters['tools'])/name)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if module.check(op,source,out/(name+'.json')):raise SystemExit('补充检查未通过')
result=engine.evaluate(request).model_dump(mode='json')
(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result.get(k) for k in ['status','geo_mean','num_passed','num_workloads','is_hack']}),flush=True)
if result['status']!='PASSED':raise SystemExit('完整评测未通过')
'''


if __name__ == '__main__':
    main()
