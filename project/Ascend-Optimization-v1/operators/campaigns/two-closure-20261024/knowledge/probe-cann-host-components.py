"""只在隔离诊断中给候选内部调用计时，定位BF16小输入的host提交开销。"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,importlib.util,json,statistics,time
import torch,torch_npu

def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

batch=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024')
out=batch/'research/cann-host-components';out.mkdir(parents=True,exist_ok=False)
source=batch/'knowledge/cumulative-cann-fifo.py'
torch.use_deterministic_algorithms(True)
oracle=load('host_oracle',batch/'catalog/ops/cumulative_trapezoid/oracle.py')
case=[json.loads(l) for l in (batch/'catalog/ops/cumulative_trapezoid/timing.jsonl').read_text().splitlines()][3]
args=oracle.gen_inputs(case,'npu:0');expected=oracle.run(**args);torch.npu.synchronize()
m=load('host_candidate',source)
for _ in range(5):m.run(**args)
torch.npu.synchronize()
raw_lib=m._library()
samples=[]
current={}

class TimedLibrary:
    def __getattr__(self,name):
        function=getattr(raw_lib,name)
        def invoke(*args):
            start=time.perf_counter()
            try:return function(*args)
            finally:current[name]=current.get(name,0)+(time.perf_counter()-start)*1000
        return invoke

class TimedKernel:
    def __init__(self,kernel):self.kernel=kernel
    def __getitem__(self,grid):
        launch=self.kernel[grid]
        def invoke(*args,**kwargs):
            start=time.perf_counter()
            try:return launch(*args,**kwargs)
            finally:current['triton_submission']=current.get('triton_submission',0)+(time.perf_counter()-start)*1000
        return invoke

proxy=TimedLibrary()
m._library=lambda:proxy
m._increments_last=TimedKernel(m._increments_last)
start=time.perf_counter()
for repeat in range(30):
    current={}
    t=time.perf_counter();got=m.run(**args);current['host_total']=(time.perf_counter()-t)*1000
    current['pending_resources']=len(m._PENDING)
    samples.append(dict(current))
torch.npu.synchronize()
elapsed=(time.perf_counter()-start)*1000/30
tol=case['tolerance'];torch.testing.assert_close(got.cpu(),expected.cpu(),atol=tol['atol'],rtol=tol['rtol'])
keys=set(k for sample in samples for k in sample)
summary={key:statistics.median(sample.get(key,0) for sample in samples) for key in sorted(keys)}
result=dict(scope='候选内部诊断包装器的host提交时间，非正式性能；不修改Torch/CANN或正式候选源码',case=case,
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),physical_device=7,correct=True,
    median_components_ms=summary,samples=samples,diagnostic_full_call_ms=elapsed)
(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['samples','case']},ensure_ascii=False),flush=True)
