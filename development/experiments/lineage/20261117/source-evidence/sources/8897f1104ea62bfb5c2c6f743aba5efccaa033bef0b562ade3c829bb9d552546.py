import hashlib,importlib.util
from pathlib import Path
import torch,torch_npu
p=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/reference-closure-20261031/references/artifacts/_hgkda_reference_closure_v3.so')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='a256a6d6f9852de865b386f321b3380528449e7262000bbd2b863f37d5491080'
s=importlib.util.spec_from_file_location('_hgkda_reference_closure_v3',p)
_cpp=importlib.util.module_from_spec(s);s.loader.exec_module(_cpp)
assert torch.__version__=='2.9.0+cpu' and torch._C._GLIBCXX_USE_CXX11_ABI==True
def run(input,weight=None,bias=None,running_mean=None,running_var=None,momentum=0.1,eps=1e-5):
    return _cpp.batch_norm(input,weight,bias,running_mean,running_var,momentum,eps)
