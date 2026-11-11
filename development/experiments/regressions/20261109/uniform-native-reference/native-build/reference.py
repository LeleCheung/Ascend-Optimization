import hashlib,importlib.util
from pathlib import Path
import torch,torch_npu
p=Path(__file__).with_name('_hgkda_cann_uniform_range_0c648cbdbec3.so')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='cb983a89607238b1686d356edb57fc1a80e8767af748d512e1f2d3d1d22ab5ca'
s=importlib.util.spec_from_file_location('_hgkda_cann_uniform_range_0c648cbdbec3',p)
_cpp=importlib.util.module_from_spec(s);s.loader.exec_module(_cpp)
assert torch.__version__=='2.9.0+cpu' and torch._C._GLIBCXX_USE_CXX11_ABI==True
def run(self,from_=0.,to=1.,*,generator=None):
    return _cpp.uniform(self,from_,to,generator)
