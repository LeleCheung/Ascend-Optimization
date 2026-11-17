import hashlib,importlib.util
from pathlib import Path
import torch,torch_npu
p=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/cann-priority-20261102/references/artifacts/_hgkda_cann_priority_v1.so')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='50c322b97e70748ff21501eedc4bbf730c6df2c9ea03871f5c26a91ac4aaed12'
s=importlib.util.spec_from_file_location('_hgkda_cann_priority_v1',p)
_cpp=importlib.util.module_from_spec(s);s.loader.exec_module(_cpp)
assert torch.__version__=='2.9.0+cpu' and torch._C._GLIBCXX_USE_CXX11_ABI==True
def run(input,output_size,align_corners=False,scales_h=None,scales_w=None):
    return _cpp.upsample(input,list(output_size),align_corners,0. if scales_h is None else scales_h,0. if scales_w is None else scales_w)
