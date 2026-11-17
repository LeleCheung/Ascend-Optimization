import torch,torch_npu,hashlib,importlib.util
from pathlib import Path
p=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/cann-priority-20261102/references/artifacts/_hgkda_cann_storage_rng_v1.so')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='5af38629d8fa1d62921d6babac9d84a87e2487caf816af857ea3b8c6a693bd55'
s=importlib.util.spec_from_file_location('_hgkda_cann_storage_rng_v1',p)
_cpp=importlib.util.module_from_spec(s);s.loader.exec_module(_cpp)
def run(self,from_=0.0,to=1.0,*,generator=None):
    return _cpp.uniform(self,from_,to,generator)
