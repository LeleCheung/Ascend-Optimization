"""官方确定性条件下，采集最终扫描候选和原参考的实际路径与编译产物。"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util,types
import torch,torch_npu
batch=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024')
tools=batch.parents[2]/'tools/profiling/hgkda'
s=importlib.util.spec_from_file_location('scan_final_observer',tools/'observe-endurance.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
original=m.compiled_kernels
def wrapped(module):
    seen=set()
    for record in original(module):
        if id(record[2]) not in seen:seen.add(id(record[2]));yield record
    underlying=types.SimpleNamespace(**{name:obj.kernel for name,obj in vars(module).items() if hasattr(obj,'kernel')})
    for record in original(underlying):
        if id(record[2]) not in seen:seen.add(id(record[2]));yield record
m.compiled_kernels=wrapped
torch.use_deterministic_algorithms(True)
for index,label in [(3,'bf16-small'),(1,'fp16-large')]:
    m.observe('cumulative_trapezoid',batch/'knowledge/cumulative-cann-launch.py',batch/'catalog/ops/cumulative_trapezoid',batch/('research/scan-final-deterministic-'+label),index)
