"""直接CANN参考；事件完成后释放描述符/工作区，调用内不做全局同步。

本包装的Python/ctypes和事件管理成本仍计入完整调用；设备执行另测。
库内实现没有源码核验，因此身份始终为CANN原生库。
"""
from collections import deque
import ctypes as C
import hashlib
from pathlib import Path

ROOT=Path('/usr/local/Ascend/cann-9.0.0')
REGISTRY={
    'log10_':dict(header='aclnnop/aclnn_log10.h',query='aclnnInplaceLog10GetWorkspaceSize',execute='aclnnInplaceLog10',kind='inplace'),
    'hardswish_out':dict(header='aclnnop/aclnn_hardswish.h',query='aclnnHardswishGetWorkspaceSize',execute='aclnnHardswish',kind='out'),
    'native_group_norm':dict(header='aclnnop/aclnn_group_norm.h',query='aclnnGroupNormGetWorkspaceSize',execute='aclnnGroupNorm',kind='group_norm'),
    'expm1_':dict(header='aclnnop/aclnn_expm1.h',query='aclnnInplaceExpm1GetWorkspaceSize',execute='aclnnInplaceExpm1',kind='inplace'),
}


class Lifetimes:
    """只回收已完成事件；不同stream的资源可独立完成，不等待队头。"""
    def __init__(self,destroy):
        self.destroy=destroy
        self.pending=deque()

    def retain(self,event,pointers,objects):
        self.pending.append((event,list(pointers),objects))

    def collect(self):
        retained=deque()
        # query异常时先保留尚未处理的全部资源，再向调用者报告。
        while self.pending:
            entry=self.pending[0]
            try:
                completed=entry[0].query()
            except Exception:
                retained.extend(self.pending)
                self.pending=retained
                raise
            self.pending.popleft()
            if completed:
                for pointer in entry[1]:self.destroy(pointer)
            else:
                retained.append(entry)
        self.pending=retained

    def finish(self):
        for event,_,_ in self.pending:event.synchronize()
        self.collect()


class Native:
    def __init__(self,operator):
        import torch_npu
        self.operator=operator;self.record=REGISTRY[operator]
        self.raw_stream=getattr(torch_npu._C,'_npu_getCurrentRawStream',None)
        self.api=C.CDLL(str(ROOT/'lib64/libopapi.so'),mode=C.RTLD_GLOBAL)
        self.acl=C.CDLL(str(ROOT/'lib64/libascendcl.so'),mode=C.RTLD_GLOBAL)
        self.create=self.api.aclCreateTensor
        self.create.argtypes=[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p]
        self.create.restype=C.c_void_p
        self.destroy=self.api.aclDestroyTensor
        self.destroy.argtypes=[C.c_void_p];self.destroy.restype=C.c_int
        self.query=getattr(self.api,self.record['query']);self.query.restype=C.c_int
        kind=self.record['kind']
        self.query.argtypes=([C.c_void_p] if kind=='inplace' else [C.c_void_p]*2 if kind=='out' else
            [C.c_void_p]*3+[C.c_int64]*4+[C.c_double]+[C.c_void_p]*3)+[C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)]
        self.execute=getattr(self.api,self.record['execute'])
        self.execute.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p];self.execute.restype=C.c_int
        self.lifetimes=Lifetimes(self.destroy)
        self.quarantined=[]

    def tensor(self,value):
        if value is None:return None
        dtype={'torch.float32':0,'torch.float16':1,'torch.bfloat16':27}[str(value.dtype)]
        if any(s<0 for s in value.stride()):raise ValueError('未支持负stride')
        shape=(C.c_int64*value.ndim)(*value.shape)
        stride=(C.c_int64*value.ndim)(*value.stride())
        span=1+sum((n-1)*s for n,s in zip(value.shape,value.stride())) if value.numel() else 0
        storage=(C.c_int64*1)(span)
        pointer=self.create(shape,value.ndim,dtype,stride,0,2,storage,1,C.c_void_p(value.data_ptr()))
        if not pointer:raise RuntimeError('aclCreateTensor失败')
        return pointer

    def run(self,**kwargs):
        import torch
        self.lifetimes.collect()
        pointers=[]
        def wrap(value):
            pointer=self.tensor(value)
            if pointer:pointers.append(pointer)
            return pointer
        output=None;workspace=None;submitted=False;stream=None
        try:
            kind=self.record['kind']
            if kind=='inplace':
                x=kwargs['A'];output=x;args=[wrap(x)]
                if self.operator=='expm1_' and x.dtype==torch.bfloat16:
                    raise ValueError('本机原地expm1未核验BF16支持')
            elif kind=='out':
                x=kwargs['x'];output=kwargs['out'];args=[wrap(x),wrap(output)]
            else:
                x=kwargs['input'];n=kwargs['N'];g=kwargs['group']
                output=(torch.empty(x.shape,device=x.device,dtype=x.dtype),
                        torch.empty((n,g),device=x.device,dtype=x.dtype),
                        torch.empty((n,g),device=x.device,dtype=x.dtype))
                args=[wrap(x),wrap(kwargs['weight']),wrap(kwargs['bias']),n,kwargs['C'],kwargs['HxW'],g,kwargs['eps'],*[wrap(v) for v in output]]
            size=C.c_uint64();executor=C.c_void_p()
            code=self.query(*args,C.byref(size),C.byref(executor))
            if code:raise RuntimeError(self.record['query']+' rc='+str(code))
            workspace=torch.empty((size.value,),device=x.device,dtype=torch.uint8) if size.value else None
            stream=torch.npu.current_stream(x.device)
            # 先提交PyTorch宿主task queue中的前序操作；这不等待设备完成。
            # NoWait getter不能用于跨PyTorch/aclnn的直接runtime调用。
            if self.raw_stream is None:
                raise RuntimeError('缺少能刷新宿主task queue的stream接口，不允许猜测跨框架顺序')
            raw_stream=self.raw_stream(x.device.index)
            # 进入execute后，即使接口返回错误，也不能假定没有部分工作入队。
            submitted=True
            code=self.execute(C.c_void_p(workspace.data_ptr()) if workspace is not None else None,
                              size.value,executor,C.c_void_p(raw_stream))
            if code:raise RuntimeError(self.record['execute']+' rc='+str(code))
            event=torch.npu.Event(enable_timing=False)
            event.record(stream)
            self.lifetimes.retain(event,pointers,(kwargs,output,workspace,executor))
            pointers=[]
            return output
        except Exception:
            if submitted:
                # 只有异常路径允许同步；同步失败则保留资源，不提前释放。
                try:
                    stream.synchronize()
                except Exception:
                    self.quarantined.append((pointers,kwargs,output,workspace))
                    pointers=[]
            raise
        finally:
            for pointer in pointers:self.destroy(pointer)

    def finish(self):
        self.lifetimes.finish()

    def provenance(self):
        header=ROOT/'include'/self.record['header']
        library=(ROOT/'lib64/libopapi.so').resolve()
        return dict(operator=self.operator,reference_kind='CANN_native_aclnn',**self.record,
                    header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
                    library=str(library),library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                    binding_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    lifecycle='每次执行后在当前stream记录事件；事件完成后释放资源，无正常路径全局同步',
                    timing_limitation='完整调用包含Python ctypes及事件管理开销；不代表最佳C++宿主封装',
                    source_implementation='库内实现源码未核验')

_NATIVE=Native('hardswish_out')
def run(x, out):
    return _NATIVE.run(x=x,out=out)
