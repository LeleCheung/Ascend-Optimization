import hashlib,importlib.util
from pathlib import Path
import torch,torch_npu
p=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/cann-priority-20261102/references/artifacts/_hgkda_cann_priority_v1.so')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='50c322b97e70748ff21501eedc4bbf730c6df2c9ea03871f5c26a91ac4aaed12'
s=importlib.util.spec_from_file_location('_hgkda_cann_priority_v1',p)
_cpp=importlib.util.module_from_spec(s);s.loader.exec_module(_cpp)
assert torch.__version__=='2.9.0+cpu' and torch._C._GLIBCXX_USE_CXX11_ABI==True
def run(qkv,qkv_bias,num_heads):
    return _cpp.qkv(qkv,qkv_bias,num_heads)
