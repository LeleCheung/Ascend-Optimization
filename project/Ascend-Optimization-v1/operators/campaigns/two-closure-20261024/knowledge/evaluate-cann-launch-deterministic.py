"""使用原KGS引擎，单列官方确定性条件的完整评测，不更改服务或原合同。"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib, importlib.util, json
import torch, torch_npu
from kernelgen_client.catalog import Catalog
from kernelgen_client.protocol.schema import Implementation, EvaluationSettings
from kernelgen_server.evaluation.engine import EvaluationEngine
from kernelgen_server.backends.ascend.device import NpuDevice

batch=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024')
source=batch/'knowledge/cumulative-cann-launch.py'
out=batch/'research/cann-launch-deterministic-complete'
out.mkdir(parents=True,exist_ok=False)
torch.use_deterministic_algorithms(True)
torch.npu.set_device('npu:0')
catalog=Catalog(batch/'catalog')
operator=catalog.load('cumulative_trapezoid')
implementation=Implementation(name='cumulative-trapezoid-cann-launch-deterministic',
    definition='cumulative_trapezoid',language='triton',entrypoint='main.py::run',
    sources=[dict(path='main.py',content=source.read_text())])
settings=EvaluationSettings(warmup_ms=1000,benchmark_ms=100,num_trials=3,timeout_seconds=1800)
request=catalog.request('cumulative_trapezoid',implementation,settings)
(out/'request.json').write_text(json.dumps(request.model_dump(mode='json'),indent=2)+'\n')
meta=dict(scope='原KGS隔离引擎完整评测；官方确定性条件另列，非默认合同交付',
    deterministic=True,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    contract_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (batch/'catalog/ops/cumulative_trapezoid').iterdir() if p.is_file()},
    torch_version=torch.__version__,torch_npu_version=torch_npu.__version__,physical_device=7)
(out/'conditions.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
engine=EvaluationEngine(NpuDevice(perf_mode='walltime',num_trials=3),'npu:0','ascend',oracle_path=operator.oracle_path)
preflight=engine.preflight(request)
(out/'preflight.json').write_text(json.dumps(preflight,ensure_ascii=False,indent=2)+'\n')
print('PREFLIGHT',json.dumps(preflight,ensure_ascii=False)[-3000:],flush=True)
if preflight.get('status')!='PASSED':
    raise SystemExit('原KGS准入未通过；不绕过准入')
tools=batch.parents[2]/'tools/profiling/hgkda'
s=importlib.util.spec_from_file_location('deterministic_gate',tools/'check-endurance.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
gate_code=m.check('cumulative_trapezoid',source,out/'semantics.json')
print('GATE_EXIT',gate_code,flush=True)
if gate_code: raise SystemExit('额外语义未通过，不评测')
result=engine.evaluate(request)
raw=result.model_dump(mode='json')
(out/'result.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2)+'\n')
print('DETERMINISTIC_COMPLETE',json.dumps({key:raw.get(key) for key in ['status','geo_mean','num_passed','num_workloads','is_hack','hack_reason']}),flush=True)
