"""直接调用CANN aclnn；PyTorch只提供张量/stream，不计算目标算子。"""
import ctypes as C
import hashlib
from pathlib import Path

ROOT=Path('/usr/local/Ascend/cann-9.0.0')
REGISTRY={
    'expm1_':dict(reference_kind='CANN_native_aclnn',header='aclnnop/aclnn_expm1.h',query='aclnnInplaceExpm1GetWorkspaceSize',execute='aclnnInplaceExpm1',dtypes=['float16','float32'],effects='原地修改并返回同一张量',source_implementation='库内实现未核验，不标作已获得Ascend C源码'),
    'native_group_norm':dict(reference_kind='CANN_native_aclnn',header='aclnnop/aclnn_group_norm.h',query='aclnnGroupNormGetWorkspaceSize',execute='aclnnGroupNorm',dtypes=['float16','bfloat16','float32'],effects='输入不变，返回out/mean/rstd',source_implementation='库内实现未核验，不标作已获得Ascend C源码'),
}


class Native:
    def __init__(self, operator):
        self.operator=operator;self.record=REGISTRY[operator]
        self.api=C.CDLL(str(ROOT/'lib64/libopapi.so'),mode=C.RTLD_GLOBAL)
        self.acl=C.CDLL(str(ROOT/'lib64/libascendcl.so'),mode=C.RTLD_GLOBAL)
        self.create=self.api.aclCreateTensor
        self.create.argtypes=[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p]
        self.create.restype=C.c_void_p
        self.destroy=self.api.aclDestroyTensor;self.destroy.argtypes=[C.c_void_p];self.destroy.restype=C.c_int
        self.query=getattr(self.api,self.record['query']);self.execute=getattr(self.api,self.record['execute'])
        self.execute.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p];self.execute.restype=C.c_int
        if operator=='expm1_':self.query.argtypes=[C.c_void_p,C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)]
        else:self.query.argtypes=[C.c_void_p]*3+[C.c_int64]*4+[C.c_double]+[C.c_void_p]*3+[C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)]
        self.query.restype=C.c_int

    def tensor(self, value):
        if value is None:return None
        dtype={'torch.float32':0,'torch.float16':1,'torch.bfloat16':27}[str(value.dtype)]
        shape=(C.c_int64*value.ndim)(*value.shape);stride=(C.c_int64*value.ndim)(*value.stride())
        # data_ptr已经包含storage_offset；使用从该地址起的物理范围且offset=0。
        span=1+sum((n-1)*s for n,s in zip(value.shape,value.stride())) if value.numel() else 0
        storage=(C.c_int64*1)(span)
        tensor=self.create(shape,value.ndim,dtype,stride,0,2,storage,1,C.c_void_p(value.data_ptr()))
        if not tensor:raise RuntimeError('aclCreateTensor失败')
        return tensor

    def run(self, **kwargs):
        import torch
        pointers=[];workspace=None
        def wrap(value):
            p=self.tensor(value)
            if p:pointers.append(p)
            return p
        try:
            if self.operator=='expm1_':
                x=kwargs['A']
                if str(x.dtype).split('.')[-1] not in self.record['dtypes']:raise ValueError('该CANN原地接口未声明此dtype')
                output=x;args=[wrap(x)]
            else:
                x=kwargs['input'];n=kwargs['N'];g=kwargs['group']
                output=(torch.empty_like(x),torch.empty((n,g),device=x.device,dtype=x.dtype),torch.empty((n,g),device=x.device,dtype=x.dtype))
                args=[wrap(x),wrap(kwargs['weight']),wrap(kwargs['bias']),n,kwargs['C'],kwargs['HxW'],g,kwargs['eps'],*[wrap(v) for v in output]]
            size=C.c_uint64();executor=C.c_void_p()
            code=self.query(*args,C.byref(size),C.byref(executor))
            if code:raise RuntimeError(self.record['query']+' rc='+str(code))
            workspace=torch.empty((size.value,),device=x.device,dtype=torch.uint8) if size.value else None
            stream=torch.npu.current_stream().npu_stream
            code=self.execute(C.c_void_p(workspace.data_ptr()) if workspace is not None else None,size.value,executor,C.c_void_p(stream))
            if code:raise RuntimeError(self.record['execute']+' rc='+str(code))
            # 本参考验证入口显式同步，保证descriptor/workspace生命周期；计时scope必须包含此同步或另行保留执行器。
            torch.npu.synchronize()
            return output
        finally:
            for pointer in pointers:self.destroy(pointer)

    def provenance(self):
        header=ROOT/'include'/self.record['header'];library=(ROOT/'lib64/libopapi.so').resolve()
        metadata=dict(self.record);metadata.pop('header')
        return dict(operator=self.operator,**metadata,header=str(header),header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
                    library=str(library),library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                    binding_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    scope='直接aclnn两段接口；PyTorch仅分配张量/获取stream，生命周期同步导致不可直接作为完整调用速度成绩')
