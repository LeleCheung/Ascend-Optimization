import hashlib,importlib.util
from pathlib import Path
import torch,torch_npu
p=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/reference-closure-20261031/references/artifacts/_hgkda_reference_closure_v2.so')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='1cc96d58b508640b344f9fa2e64a14f0ba63babc28e7f2ec82a688cfa420f096'
s=importlib.util.spec_from_file_location('_hgkda_reference_closure_v2',p)
_cpp=importlib.util.module_from_spec(s);s.loader.exec_module(_cpp)
assert torch.__version__=='2.9.0+cpu' and torch._C._GLIBCXX_USE_CXX11_ABI==True
def run(input,weight=None,bias=None,running_mean=None,running_var=None,momentum=0.1,eps=1e-5):
    return _cpp.batch_norm(input,weight,bias,running_mean,running_var,momentum,eps)
