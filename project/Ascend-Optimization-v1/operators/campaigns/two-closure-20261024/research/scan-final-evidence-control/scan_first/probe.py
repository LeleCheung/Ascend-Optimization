"""新进程先执行候选再执行参考，检查确定性是否依赖参考先初始化运行时。"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ctypes,hashlib,importlib.util,json
import numpy as np
import torch,torch_npu

def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

batch=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024')
out=batch/'research/launch-candidate-first-deterministic';out.mkdir(parents=True,exist_ok=False)
source=batch/'knowledge/cumulative-cann-launch.py'
torch.use_deterministic_algorithms(True)
torch.npu.set_device('npu:0')
oracle=load('first_oracle',batch/'catalog/ops/cumulative_trapezoid/oracle.py')
case=[json.loads(l) for l in (batch/'catalog/ops/cumulative_trapezoid/timing.jsonl').read_text().splitlines()][2]
args=oracle.gen_inputs(case,'npu:0');torch.npu.synchronize()
lib=ctypes.CDLL('libacl_rt.so')
lib.aclrtGetSysParamOpt.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_int64)]
lib.aclrtGetSysParamOpt.restype=ctypes.c_int
def runtime_option():
    value=ctypes.c_int64(-1);code=lib.aclrtGetSysParamOpt(0,ctypes.byref(value));return dict(returncode=code,value=value.value)
metadata=dict(before=runtime_option(),scope='全新隔离进程仅先设置torch官方确定性API，候选先于任何积分或cumsum参考执行',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
candidate=load('first_candidate',source)
values=[]
for _ in range(6):
    v=candidate.run(**args);torch.npu.synchronize();values.append(v.cpu().float().numpy().astype(np.float64))
stack=np.stack(values);tol=case['tolerance'];radius=tol['atol']+tol['rtol']*np.abs(stack)
metadata.update(candidate_six_bitwise_equal=bool(np.all(stack==stack[0])),
    empty_tolerance_intersections=int((np.max(stack-radius,axis=0)>np.min(stack+radius,axis=0)).sum()),
    before_reference=runtime_option())
reference=oracle.run(**args);torch.npu.synchronize()
ref=reference.cpu().float().numpy().astype(np.float64)
metadata.update(after_reference=runtime_option(),mismatch_vs_first_reference=[int((np.abs(v-ref)>tol['atol']+tol['rtol']*np.abs(ref)).sum()) for v in values])
(out/'result.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(metadata,ensure_ascii=False),flush=True)
