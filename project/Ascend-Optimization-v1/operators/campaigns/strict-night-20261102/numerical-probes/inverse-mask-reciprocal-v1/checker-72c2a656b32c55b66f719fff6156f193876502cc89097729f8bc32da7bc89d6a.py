"""把LU阶段偏差缩小为循环掩码保留及标量倒数的数值探测，不计时。"""
import argparse
import hashlib
import json
from pathlib import Path

NAME='inverse-mask-reciprocal-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process(runner):
    from night_common import PY,read
    folder=runner.batch/'numerical-probes'/NAME
    spec=folder/'spec.json'
    if not spec.exists():return
    configuration=read(spec)
    assert configuration['checker_sha256']==sha(__file__)
    assert configuration['source_sha256']==sha(configuration['source'])
    result=folder/'result.json'
    key='diagnostic:'+NAME
    previous=runner.state['jobs'].get(key,{})
    if not result.exists():
        assert previous.get('status') not in {'running','completed'},'未知回执不重发'
        frozen=folder/('checker-'+sha(__file__)+'.py')
        if frozen.exists():assert frozen.read_bytes()==Path(__file__).read_bytes()
        else:frozen.write_bytes(Path(__file__).read_bytes())
        current=runner.state.get('current_job')
        runner.state['current_job']=key
        runner.state['jobs'][key]=dict(status='running',checker=str(frozen),spec_sha256=sha(spec))
        runner.persist()
        runner.cmd(key,[PY,frozen,'--spec',spec,'--output',result],900)
        runner.state['current_job']=current
    value=read(result)
    assert value['checker_sha256']==configuration['checker_sha256']
    assert value['source_sha256']==configuration['source_sha256']
    assert value['spec_sha256']==sha(spec)
    assert value['performance_measured'] is False and value['qualified_CANN'] is False
    runner.state['jobs'][key]=dict(status='completed',result=str(result),
        result_sha256=sha(result),outcome=value['status'])
    runner.persist()


KERNELS='''
@triton.jit(do_not_specialize=["STEPS"])
def _masked_update(A, F, V, T, OUT, STEPS, BLOCK: tl.constexpr, MODE: tl.constexpr):
    pid=tl.program_id(0)
    i=tl.arange(0,BLOCK)
    threshold=tl.load(T+pid)
    active=i>=threshold
    column=tl.load(V+pid*BLOCK+i)
    for k in tl.range(0,STEPS):
        lhs=tl.load(A+pid*STEPS*BLOCK+k*BLOCK+i)
        factor=tl.load(F+pid*STEPS+k)
        if MODE==0:
            column=tl.where(active,_compensated_fma(lhs,factor,column),column)
        elif MODE==1:
            column=_compensated_fma(tl.where(active,lhs,0.),
                                    tl.where(active,factor,0.),column)
        else:
            column=tl.where(active,column+lhs*factor,column)
    tl.store(OUT+pid*BLOCK+i,column)


@triton.jit
def _reciprocal_forms(D, T, OLD, VECTOR, BLOCK: tl.constexpr):
    pid=tl.program_id(0)
    i=tl.arange(0,BLOCK)
    denominator=tl.load(D+pid)
    active=i<tl.load(T+pid)
    one=tl.full((BLOCK,),1.,tl.float32)
    old=_corrected_div(one,denominator)
    vector=_corrected_div(one,tl.where(active,denominator,1.))
    tl.store(OLD+pid*BLOCK+i,old)
    tl.store(VECTOR+pid*BLOCK+i,vector)
'''


def measure(configuration,out):
    import ctypes
    import ctypes.util
    import importlib.util
    import struct
    import torch
    import torch_npu

    source=Path(configuration['source'])
    text=source.read_text(encoding='utf8')
    start=text.index('@triton.jit\ndef _compensated_fma')
    end=text.index('@triton.jit\ndef _leftlooking')
    kernel=out/'kernel.py'
    assert not kernel.exists()
    kernel.write_text('import triton\nimport triton.language as tl\n\n'+text[start:end]+KERNELS,encoding='utf8')
    spec=importlib.util.spec_from_file_location('inverse_mask_reciprocal_probe',kernel)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    libm=ctypes.CDLL(ctypes.util.find_library('m'))
    fma=libm.fmaf
    fma.argtypes=[ctypes.c_float]*3
    fma.restype=ctypes.c_float
    bits=lambda x:struct.pack('<f',x)
    f32=lambda x:struct.unpack('<f',bits(x))[0]
    stage=json.loads(Path(configuration['stages']).read_text(encoding='utf8'))
    generator=torch.Generator().manual_seed(715038)
    batch,block=8,16
    thresholds=torch.tensor([2,1,3,7,5,8,2,6],dtype=torch.int32)
    modes=[]
    artifacts={}

    def freeze(compiled,label):
        paths={}
        for kind,data in compiled.asm.items():
            if not isinstance(data,(str,bytes)):continue
            path=out/(label+'.'+kind)
            path.write_bytes(data.encode() if isinstance(data,str) else data)
            paths[kind]=dict(path=str(path),sha256=sha(path))
        artifacts[label]=paths

    for steps in [2,7]:
        lhs=torch.randn((batch,steps,block),generator=generator)
        factors=torch.randn((batch,steps),generator=generator)
        initial=torch.randn((batch,block),generator=generator)
        # 重建第一组3×3的最后一列下半更新；常量只进入诊断输入，不进入候选。
        if steps==2:
            first=stage['cases'][0]
            lu=torch.tensor(first['cpu_lu'],dtype=torch.float32)
            original=torch.tensor(first['input'],dtype=torch.float32)
            lhs[0].zero_()
            initial[0].zero_()
            initial[0,:3]=torch.tensor([lu[0,2],lu[1,2],original[2,2]])
            lhs[0,0,:3]=-lu[:,0]
            lhs[0,1,:3]=-lu[:,1]
            factors[0,0]=lu[0,2]
            factors[0,1]=lu[1,2]
        payload=dict(steps=steps,lhs=lhs.tolist(),factors=factors.tolist(),
                     initial=initial.tolist(),thresholds=thresholds.tolist())
        (out/('inputs-'+str(steps)+'.json')).write_text(json.dumps(payload,indent=2)+'\n',encoding='utf8')
        device=[x.to('npu:0') for x in [lhs,factors,initial,thresholds]]
        for mode in [0,1,2]:
            actual=torch.empty_like(initial,device='npu:0')
            compiled=module._masked_update[(batch,)](*device,actual,steps,block,mode,
                                                    enable_fp_fusion=False,num_warps=32)
            torch.npu.synchronize()
            values=actual.cpu().tolist()
            inactive=[]
            active_errors=[]
            for pid in range(batch):
                for i in range(block):
                    expected=initial[pid,i].item()
                    if i>=thresholds[pid].item():
                        for k in range(steps):
                            a,b=lhs[pid,k,i].item(),factors[pid,k].item()
                            expected=fma(a,b,expected) if mode!=2 else f32(expected+f32(a*b))
                        if bits(expected)!=bits(values[pid][i]):
                            active_errors.append(dict(program=pid,index=i,expected=expected,actual=values[pid][i]))
                    elif bits(expected)!=bits(values[pid][i]):
                        inactive.append(dict(program=pid,index=i,expected=expected,actual=values[pid][i]))
            modes.append(dict(steps=steps,mode=mode,inactive_changed=len(inactive),
                active_mismatches=len(active_errors),first_inactive=inactive[:8],
                first_active_errors=active_errors[:8],actual=values))
            freeze(compiled,'masked-'+str(steps)+'-'+str(mode))
            (out/'masked-modes.json').write_text(json.dumps(modes,indent=2)+'\n',encoding='utf8')
    scalar=stage['scalar_rows']
    denominators=torch.tensor([scalar[i]['b'] for i in range(0,len(scalar),128)],dtype=torch.float32)
    old=torch.empty((batch,block),dtype=torch.float32,device='npu:0')
    vector=torch.empty_like(old)
    compiled=module._reciprocal_forms[(batch,)](denominators.to('npu:0'),thresholds.to('npu:0'),
                                              old,vector,block,enable_fp_fusion=False,num_warps=32)
    torch.npu.synchronize()
    old_values,vector_values=old.cpu().tolist(),vector.cpu().tolist()
    reciprocal=[]
    for pid in range(batch):
        target=f32(1./denominators[pid].item())
        active=thresholds[pid].item()
        reciprocal.append(dict(denominator=denominators[pid].item(),active=active,expected=target,
            old=old_values[pid][:active],vector=vector_values[pid][:active],
            old_mismatches=sum(bits(x)!=bits(target) for x in old_values[pid][:active]),
            vector_mismatches=sum(bits(x)!=bits(target) for x in vector_values[pid][:active])))
    freeze(compiled,'reciprocal-forms')
    return dict(status='measured',modes=modes,reciprocal=reciprocal,artifacts=artifacts,
        kernel_sha256=sha(kernel),stages_sha256=sha(configuration['stages']),device_executed=True,
        scope='仅验证循环掩码保留与两种倒数表达的有限输入，不计时；最小探测未复现也不能否定完整LU阶段证据。')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    configuration=json.loads(args.spec.read_text(encoding='utf8'))
    assert configuration['checker_sha256']==sha(__file__)
    assert configuration['source_sha256']==sha(configuration['source'])
    try:value=measure(configuration,args.output.parent)
    except Exception as error:
        import traceback
        value=dict(status='diagnostic_error',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
    value.update(checker_sha256=sha(__file__),spec_sha256=sha(args.spec),
        source_sha256=configuration['source_sha256'],qualified_CANN=False,performance_measured=False)
    args.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=value['status'],error=value.get('error'),
        modes=[{k:r[k] for k in ['steps','mode','inactive_changed','active_mismatches']} for r in value.get('modes',[])])))


if __name__=='__main__':main()
