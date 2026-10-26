import sys
sys.dont_write_bytecode=True
import hashlib,importlib.util,json,torch,torch_npu
from pathlib import Path
b=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024')
s=importlib.util.spec_from_file_location('stable_model_oracle',b/'catalog/ops/cumulative_trapezoid/oracle.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
cases=[json.loads(line) for line in (b/'catalog/ops/cumulative_trapezoid/timing.jsonl').read_text().splitlines()]
torch.use_deterministic_algorithms(True)
rows=[]
for case in cases:
 args=m.gen_inputs(case,'npu:0');x=args['y'];dim=args['dim'];dx=args['dx'];expected=m.run(**args);torch.npu.synchronize();cpu=expected.cpu()
 a=x.cpu();length=a.shape[dim]-1
 left=a.narrow(dim,0,length);right=a.narrow(dim,1,length)
 increment=(left+right)*(dx/2.)
 models={'fp32_prefix':increment.float().cumsum(dim).to(a.dtype),'fp64_prefix':increment.double().cumsum(dim).to(a.dtype)}
 tol=case['tolerance'];row=dict(case=case['name'],spec=case['inputs']['spec'],models={})
 for name,value in models.items():
  err=(value.double()-cpu.double()).abs();ok=err<=tol['atol']+tol['rtol']*cpu.double().abs()
  row['models'][name]=dict(passed=bool(ok.all()),mismatches=int((~ok).sum()),max_abs=float(err.max()))
 rows.append(row);print('MODEL',json.dumps(row),flush=True)
out=b/'research/stable-models';out.mkdir(parents=True,exist_ok=True)
(out/'models.json').write_text(json.dumps(dict(scope='官方确定性条件；CPU数值模型诊断，不作正式性能',rows=rows),indent=2)+'\n')
