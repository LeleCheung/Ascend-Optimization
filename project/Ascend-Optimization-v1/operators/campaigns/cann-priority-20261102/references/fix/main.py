import hashlib,importlib.util
from pathlib import Path
import torch,torch_npu
p=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/reference-closure-20261031/references/artifacts/_hgkda_reference_closure_v1.so')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='f70b1d5e4df55cae2ea8c9754c6aa95d9725f6e470cea116cdd97ee3d9fcb3d4'
s=importlib.util.spec_from_file_location('_hgkda_reference_closure_v1',p)
_cpp=importlib.util.module_from_spec(s);s.loader.exec_module(_cpp)
assert torch.__version__=='2.9.0+cpu' and torch._C._GLIBCXX_USE_CXX11_ABI==True
def run(self):
    return _cpp.trunc(self)
