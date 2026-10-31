"""受控 GEMM/流式复制驱动：保存真实编译产物，单次目标提交。"""
import argparse,hashlib,json,os
from pathlib import Path
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def roof_calibration_gemm(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                          BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    mi=tl.program_id(0)*BM+tl.arange(0,BM)
    ni=tl.program_id(1)*BN+tl.arange(0,BN)
    ki=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for start in range(tl.cdiv(K,BK)):
        ix=start*BK+ki
        a=tl.load(A+mi[:,None]*K+ix[None,:],(mi[:,None]<M)&(ix[None,:]<K),0)
        b=tl.load(B+ix[:,None]*N+ni[None,:],(ix[:,None]<K)&(ni[None,:]<N),0)
        acc+=tl.dot(a,b,allow_tf32=False)
    tl.store(C+mi[:,None]*N+ni[None,:],acc,(mi[:,None]<M)&(ni[None,:]<N))

@triton.jit
def roof_calibration_copy(A,C,S:tl.constexpr,BLOCK:tl.constexpr):
    for start in range(tl.program_id(0)*BLOCK,S,tl.num_programs(0)*BLOCK):
        idx=start+tl.arange(0,BLOCK)
        value=tl.load(A+idx,idx<S,0)
        tl.store(C+idx,value,idx<S)

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();spec=json.loads(a.spec.read_text());a.output.mkdir(parents=True,exist_ok=True)
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    torch.manual_seed(13);dtype=getattr(torch,spec.get('dtype','float16'))
    if spec['kind']=='gemm':
        m,n,k=spec['shape'];x=torch.randn((m,k),dtype=dtype,device='npu');y=torch.randn((k,n),dtype=dtype,device='npu')
        c=torch.empty((m,n),dtype=dtype,device='npu');bm,bn,bk=spec['tile']
        def call():return roof_calibration_gemm[(triton.cdiv(m,bm),triton.cdiv(n,bn))](x,y,c,m,n,k,bm,bn,bk)
    else:
        size=spec['elements'];x=torch.ones(size,dtype=dtype,device='npu');c=torch.empty_like(x)
        def call():return roof_calibration_copy[(min(spec.get('programs',40),triton.cdiv(size,spec.get('block',4096))),)](x,c,size,spec.get('block',4096))
    compiled=call();torch.npu.synchronize()
    artifacts=[]
    for kind,data in compiled.asm.items():
        data=data if isinstance(data,bytes) else str(data).encode()
        path=a.output/(kind+('.bin' if isinstance(compiled.asm[kind],bytes) else '.txt'))
        if path.exists():assert path.read_bytes()==data
        else:path.write_bytes(data)
        artifacts.append(dict(path=path.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    if spec.get('validate',True):
        reference=x@y if spec['kind']=='gemm' else x
        if spec['kind']=='gemm':torch.testing.assert_close(c,reference,rtol=1e-2,atol=1e-2)
        else:torch.testing.assert_close(c,reference,rtol=0,atol=0)
    record=dict(spec=spec,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                artifacts=artifacts,metadata=str(compiled.metadata),status='passed',device_name=torch.npu.get_device_name())
    (a.output/'driver.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(status='passed',kind=spec['kind'],spec=spec)),flush=True)

if __name__=='__main__':main()
