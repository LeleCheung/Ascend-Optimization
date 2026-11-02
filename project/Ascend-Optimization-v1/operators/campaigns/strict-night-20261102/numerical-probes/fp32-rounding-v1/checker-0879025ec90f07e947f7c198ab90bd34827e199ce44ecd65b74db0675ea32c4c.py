"""串行诊断FP32融合、拆分、补偿乘加与除法；不是性能评测或候选交付。"""
import argparse
import hashlib
import json
from pathlib import Path


ARITHMETIC_SOURCE = """import triton
import triton.language as tl

@triton.jit
def arithmetic(A,B,C,F,S,H,D,Q,N:tl.constexpr):
    i=tl.program_id(0)*256+tl.arange(0,256)
    a=tl.load(A+i,i<N,other=1.)
    b=tl.load(B+i,i<N,other=1.)
    c=tl.load(C+i,i<N,other=0.)
    product=a*b
    split=product+c
    mask=tl.full((),0xfffff000,tl.uint32)
    ah=(a.to(tl.uint32,bitcast=True)&mask).to(tl.float32,bitcast=True)
    bh=(b.to(tl.uint32,bitcast=True)&mask).to(tl.float32,bitcast=True)
    al=a-ah
    bl=b-bh
    e1=product-ah*bh
    e2=e1-al*bh
    e3=e2-ah*bl
    product_error=al*bl-e3
    virtual=split-product
    sum_error=(product-(split-virtual))+(c-virtual)
    compensated=split+(sum_error+product_error)
    tl.store(F+i,tl.fma(a,b,c),i<N)
    tl.store(S+i,split,i<N)
    tl.store(H+i,compensated,i<N)
    tl.store(D+i,tl.div_rn(a,b),i<N)
    tl.store(Q+i,a/b,i<N)

"""

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process(runner):
    from night_common import PY,read,save
    out=runner.batch/'numerical-probes/fp32-rounding-v1';spec=out/'spec.json'
    if not spec.exists():return
    configuration=read(spec)
    assert configuration['checker_sha256']==sha(__file__),'诊断源码与排队身份不同'
    result=out/'result.json';key='diagnostic:fp32-rounding-v1'
    previous=runner.state['jobs'].get(key,{})
    if not result.exists():
        assert previous.get('status')!='running','诊断回执未知，不重新提交'
        assert previous.get('status')!='completed','完成记录缺少结果，先检查'
        frozen=out/('checker-'+sha(__file__)+'.py')
        if frozen.exists():assert frozen.read_bytes()==Path(__file__).read_bytes()
        else:frozen.write_bytes(Path(__file__).read_bytes())
        original=runner.state.get('current_job')
        runner.state['current_job']=key
        runner.state['jobs'][key]=dict(status='running',checker=str(frozen),spec=str(spec),spec_sha256=sha(spec))
        runner.persist()
        runner.cmd(key,[PY,frozen,'--spec',spec,'--output',result],900)
        runner.state['current_job']=original
    value=read(result)
    assert value['checker_sha256']==configuration['checker_sha256'] and value['spec_sha256']==sha(spec)
    assert value['qualified_CANN'] is False and value['performance_measured'] is False
    runner.state['jobs'][key]=dict(status='completed',result=str(result),result_sha256=sha(result),outcome=value['status'])
    runner.persist()


def measure(out):
    import ctypes
    import ctypes.util
    import struct
    import importlib.util
    import torch
    import torch_npu
    import triton

    kernel_path=out.parent/'kernel.py'
    if kernel_path.exists():assert kernel_path.read_text(encoding='utf8')==ARITHMETIC_SOURCE
    else:kernel_path.write_text(ARITHMETIC_SOURCE,encoding='utf8')
    spec=importlib.util.spec_from_file_location('hgkda_fp32_rounding_probe_kernel',kernel_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    arithmetic=module.arithmetic

    count=2048;generator=torch.Generator().manual_seed(937102)
    a=torch.randn(count,generator=generator,dtype=torch.float32)
    b=torch.randn(count,generator=generator,dtype=torch.float32)
    b=torch.where(b.abs()<0.01,torch.full_like(b,.01),b)
    c=torch.randn(count,generator=generator,dtype=torch.float32)
    c[:count//2]=-(a[:count//2]*b[:count//2])
    a[0]=1.+2.**-23;b[0]=1.-2.**-23;c[0]=-1.
    cpu=[value.tolist() for value in [a,b,c]]
    raw=struct.pack('<'+str(3*count)+'f',*(cpu[0]+cpu[1]+cpu[2]))
    inputs=out.parent/'input-fp32.bin';assert not inputs.exists();inputs.write_bytes(raw)
    device=[value.to('npu:0') for value in [a,b,c]]
    outputs=[torch.empty((count,),dtype=torch.float32,device='npu:0') for _ in range(5)]
    arithmetic[(triton.cdiv(count,256),)](*device,*outputs,count,enable_fp_fusion=False)
    torch.npu.synchronize();values=[value.cpu().tolist() for value in outputs]
    libm=ctypes.CDLL(ctypes.util.find_library('m'))
    fused=libm.fmaf;fused.argtypes=[ctypes.c_float]*3;fused.restype=ctypes.c_float
    def bits(x):return struct.pack('<f',x)
    def f32(x):return struct.unpack('<f',bits(x))[0]
    rows=[];errors={name:0 for name in ['fma','split','compensated','div_rn','ordinary_div']}
    for i,(x,y,z) in enumerate(zip(*cpu)):
        expected=fused(x,y,z);division=f32(x/y)
        row=dict(index=i,a=x,b=y,c=z,expected_fma=expected,expected_div=division)
        for name,data in zip(errors,values):
            actual=data[i];target=division if name in {'div_rn','ordinary_div'} else expected
            row[name]=actual;row[name+'_exact']=bits(actual)==bits(target)
            errors[name]+=not row[name+'_exact']
        rows.append(row)
    return dict(status='measured',samples=count,exact_mismatches=errors,rows=rows,
        input=str(inputs),input_sha256=sha(inputs),kernel=str(kernel_path),kernel_sha256=sha(kernel_path),torch=torch.__version__,torch_npu=torch_npu.__version__,
        device=str(torch.npu.get_device_name(0)),
        scope='普通FP32值与乘积消去样本；比较CPU libm fmaf的单次舍入。不证明全IEEE范围，未计时，也未改变任何候选或原评测合同。')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists(),'已有诊断不覆盖或重发'
    configuration=json.loads(args.spec.read_text(encoding='utf8'))
    assert configuration['checker_sha256']==sha(__file__)
    try:value=measure(args.output)
    except Exception as error:value=dict(status='diagnostic_error',error=type(error).__name__+': '+str(error)[:4000])
    value.update(checker_sha256=sha(__file__),spec_sha256=sha(args.spec),qualified_CANN=False,performance_measured=False)
    args.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({key:value[key] for key in ['status','exact_mismatches','error'] if key in value}))


if __name__=='__main__':main()
