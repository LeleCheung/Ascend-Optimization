import torch,torch_npu,hashlib,importlib.util
from pathlib import Path
p=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/cann-priority-20261102/references/artifacts/_hgkda_cann_priority_extra_v1.so')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='cbaf49151d3d74763a1b809abda3d4d6180bc00877f7b940c47e763ab5c6175d'
s=importlib.util.spec_from_file_location('_hgkda_cann_priority_extra_v1',p)
_cpp=importlib.util.module_from_spec(s);s.loader.exec_module(_cpp)
def run(y,*,dx=1,dim=-1):
    return _cpp.trapezoid(y,dx,dim)
