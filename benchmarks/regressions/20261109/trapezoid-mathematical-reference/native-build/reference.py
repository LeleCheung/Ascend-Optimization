import hashlib,importlib.util
from pathlib import Path
import torch,torch_npu
p=Path(__file__).with_name('_hgkda_cann_trapezoid_67b8cacf2b3b.so')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='eec7b374898cfeebc58e2efc61d6d41f59907ddd048e7b73df5e80c538b9d745'
assert torch.__version__=='2.9.0+cpu'
s=importlib.util.spec_from_file_location('_hgkda_cann_trapezoid_67b8cacf2b3b',p)
_cpp=importlib.util.module_from_spec(s);s.loader.exec_module(_cpp)
def run(y,*,dx=1,dim=-1):
    return _cpp.trapezoid(y,float(dx),int(dim))
