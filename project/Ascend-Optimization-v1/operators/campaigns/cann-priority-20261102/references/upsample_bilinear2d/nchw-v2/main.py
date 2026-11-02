import torch,torch_npu,hashlib,importlib.util
from pathlib import Path
p=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/cann-priority-20261102/references/upsample_bilinear2d/nchw-v2/_hgkda_upsample_nchw_v2.so')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='57db7e880d5a70df6baf4cd101b98a8eae6699f6d990e41d14097690630e4fe4'
s=importlib.util.spec_from_file_location("_hgkda_upsample_nchw_v2",p)
_cpp=importlib.util.module_from_spec(s);s.loader.exec_module(_cpp)
def run(input,output_size,align_corners=False,scales_h=None,scales_w=None):
    return _cpp.run(input,list(output_size),align_corners,0. if scales_h is None else scales_h,0. if scales_w is None else scales_w)
