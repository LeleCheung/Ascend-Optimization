import hashlib,importlib.util,sys
from pathlib import Path
import torch,torch_npu
_BINARY=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/threeway-20261030/references/cpp-artifacts/_hgkda_cann_cpp_norm_v2.so')
if hashlib.sha256(_BINARY.read_bytes()).hexdigest()!='d4718082b9ae80b217d9c8ed0973b0dc5fe79eeb5db1a126e06f37c1bc06ce8e':
    raise RuntimeError('CANN C++参考二进制身份不符')
if torch.__version__!='2.9.0+cpu' or torch._C._GLIBCXX_USE_CXX11_ABI!=True:
    raise RuntimeError('CANN C++参考的Torch版本或ABI不符')
_name='_hgkda_cann_cpp_norm_v2'
_spec=importlib.util.spec_from_file_location(_name,_BINARY)
_cpp=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_cpp)

def run(input,weight,bias,N,C,HxW,group,eps):
    return _cpp.group_norm(input,weight,bias,N,C,HxW,group,eps)
