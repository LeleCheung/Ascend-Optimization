"""来源和限制显式的Ascend C源码补充参考；不替换厂商库或PyTorch正式基线。"""
import ctypes
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent


class Negative:
    def __init__(self, library):
        self.library=Path(library).resolve()
        self.handle=ctypes.CDLL(str(self.library))
        self.launch=self.handle.hgkda_negative_launch
        self.launch.argtypes=[ctypes.c_uint32,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint32]
        self.launch.restype=None

    def run(self,A):
        import torch
        if A.dtype!=torch.float32 or not A.is_contiguous() or A.numel()%8 or A.numel()>=2**32:
            raise ValueError('源码参考只声明连续FP32、8对齐、uint32元素数，其他规格不支持')
        if A.numel():self.launch(min(40,(A.numel()+2047)//2048),torch.npu.current_stream().npu_stream,A.data_ptr(),A.numel())
        return A

    def provenance(self):
        files=[HERE/'ascendc/negative.cpp',HERE/'ascendc/CMakeLists.txt',self.library]
        return dict(reference_kind='source_built_AscendC_supplement',entry='hgkda_negative_fp32',
                    files={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                    qualification='自产补充参考，仅连续FP32且元素数8对齐；不代表厂商最优，不替换KGS历史成绩')
