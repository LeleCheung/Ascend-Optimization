"""独立补测当前合同的原地/out、布局、特殊值；不修改正式合同容差。"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode=True
CHECKER_BYTES=Path(__file__).read_bytes()
DEFAULT_UNIFORM_CHECKER=Path(__file__).resolve().parent.parent/'check-endurance.py'


def storage_guards(inputs):
    """为小型非连续视图的非目标存储设哨兵，检查没有误写邻接元素。"""
    import torch
    guards=[]
    for name,tensor in inputs.items():
        if not isinstance(tensor,torch.Tensor) or tensor.is_contiguous() or tensor.numel()>65536:
            continue
        size=tensor.untyped_storage().nbytes()//tensor.element_size()
        if size>262144 or any(s<0 for s in tensor.stride()):continue
        indices=torch.full(tuple(tensor.shape),tensor.storage_offset(),dtype=torch.int64)
        for dim,(n,stride) in enumerate(zip(tensor.shape,tensor.stride())):
            shape=[1]*tensor.ndim;shape[dim]=n
            indices+=torch.arange(n).reshape(shape)*stride
        occupied=torch.zeros(size,dtype=torch.bool);occupied[indices.flatten()]=True
        gap=(~occupied).to(tensor.device)
        if not gap.any().item():continue
        storage=torch.as_strided(tensor,(size,),(1,),storage_offset=0)
        storage[gap]=True if tensor.dtype==torch.bool else 37
        guards.append((name,storage,gap,storage[gap].clone()))
    return guards


def imported(name, path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['operator','leaf','source','output']:
        p.add_argument('--'+name,required=True,type=str if name=='operator' else Path)
    p.add_argument('--uniform-checker',type=Path,default=DEFAULT_UNIFORM_CHECKER)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    import torch
    import torch_npu
    oracle=imported('next_oracle',a.leaf/'oracle.py');candidate=imported('next_candidate',a.source)
    definition=json.loads((a.leaf/'definition.json').read_text())
    aliases=definition.get('effects',{}).get('returns_alias_of',{})
    rows=[]
    cases=[json.loads(line) for line in (a.leaf/'correctness.jsonl').read_text().splitlines() if line.strip()]
    for case in cases:
        row=dict(case=case['name'],passed=False)
        try:
            # 重新构建相同随机输入，保持非连续布局，不用clone抹去stride。
            ref_inputs=oracle.gen_inputs(case,'npu:0');inputs=oracle.gen_inputs(case,'npu:0')
            storage_guards(ref_inputs)
            guards=storage_guards(inputs)
            expected=oracle.run(**ref_inputs);actual=candidate.run(**inputs);torch.npu.synchronize()
            for name,storage,gap,before in guards:
                assert torch.equal(storage[gap],before),'非连续视图误写邻接存储 '+name
            row['storage_guards']=len(guards)
            if hasattr(oracle,'valid'):
                outputs=list(actual) if isinstance(actual,(list,tuple)) else [actual]
                references=list(expected) if isinstance(expected,(list,tuple)) else [expected]
                checked=oracle.valid(references,outputs,inputs,case)
                assert checked['passed'],checked.get('message')
                row['passed']=True;rows.append(row);continue
            refs=list(expected) if isinstance(expected,(list,tuple)) else [expected]
            outputs=list(actual) if isinstance(actual,(list,tuple)) else [actual]
            assert len(refs)==len(outputs)
            for ref,out in zip(refs,outputs):
                assert ref.shape==out.shape and ref.dtype==out.dtype
                tol=case['tolerance'];torch.testing.assert_close(out,ref,atol=tol['atol'],rtol=tol['rtol'],equal_nan=True)
            for output,param in aliases.items():
                out=outputs[definition['outputs'].index(output)]
                assert torch._C._is_alias_of(out,inputs[param]),'返回别名错误 '+param
                torch.testing.assert_close(inputs[param],ref_inputs[param],atol=case['tolerance']['atol'],rtol=case['tolerance']['rtol'],equal_nan=True)
            row['passed']=True
        except Exception as e:row['error']=type(e).__name__+': '+str(e)[:2000]
        rows.append(row)
    # 同一shape换数据和stream，防止缓存源码正确却复用旧张量或旧stream。
    # 随机算子另走自身状态/统计合同，不逐值套用这一门禁。
    if not hasattr(oracle,'valid') and cases:
        import copy
        streams=[torch.npu.current_stream(),torch.npu.Stream()]
        representative={}
        for case in cases:
            spec=case.get('inputs',{}).get('spec',{})
            key=(spec.get('dtype','unknown'),bool(spec.get('strided')),spec.get('affine',True))
            representative.setdefault(key,case)
        timing=[json.loads(line) for line in (a.leaf/'timing.jsonl').read_text().splitlines() if line.strip()]
        for case in reversed(timing):
            spec=case.get('inputs',{}).get('spec',{})
            representative.setdefault((spec.get('dtype','unknown'),'large',True),case)
        for case in representative.values():
            for index,stream in enumerate(streams):
                row=dict(case=case['name']+'-fresh-stream-'+str(index),passed=False,check='fresh_input_and_stream')
                try:
                    fresh=copy.deepcopy(case);fresh['seed']=fresh.get('seed',0)+100003+index
                    with torch.npu.stream(stream):
                        ref_inputs=oracle.gen_inputs(fresh,'npu:0');inputs=oracle.gen_inputs(fresh,'npu:0')
                        expected=oracle.run(**ref_inputs);actual=candidate.run(**inputs)
                        refs=list(expected) if isinstance(expected,(list,tuple)) else [expected]
                        outputs=list(actual) if isinstance(actual,(list,tuple)) else [actual]
                        assert len(refs)==len(outputs)
                        # 拷贝在调用者stream排队，若候选投到旧stream，不由全局同步掩盖顺序问题。
                        snapshots=[out.clone() for out in outputs]
                        expected_snapshots=[ref.clone() for ref in refs]
                    stream.synchronize()
                    for ref,out in zip(expected_snapshots,snapshots):
                        tol=fresh['tolerance'];torch.testing.assert_close(out,ref,atol=tol['atol'],rtol=tol['rtol'],equal_nan=True)
                    row['passed']=True
                except Exception as error:row['error']=type(error).__name__+': '+str(error)[:2000]
                rows.append(row)
    if a.operator=='cross_out':
        layouts=[('default-first-axis',(3,4,3),(3,4,3),None),
                 ('broadcast-middle',(2,3,5),(1,3,1),1),
                 ('strided-out-last',(2,4,3),(2,4,3),-1),
                 ('alias-out-middle',(2,3,5),(2,3,5),1),
                 ('alias-out-last',(2,4,3),(2,4,3),-1)]
        for dtype in [torch.float16,torch.bfloat16,torch.float32]:
            tol=next(c['tolerance'] for c in cases if c['inputs']['spec']['dtype']==str(dtype).split('.')[-1])
            for kind,xshape,yshape,dim in layouts:
                row=dict(case='cross-layout-'+str(dtype)+'-'+kind,check='cross_dimension_broadcast_and_out',passed=False)
                try:
                    generator=torch.Generator().manual_seed(110719)
                    cpu_x=torch.randn(xshape,generator=generator).to(dtype)
                    cpu_y=torch.randn(yshape,generator=generator).to(dtype)
                    x=cpu_x.to('npu:0');y=cpu_y.to('npu:0')
                    ref_x=cpu_x.to('npu:0');ref_y=cpu_y.to('npu:0')
                    if kind.startswith('alias-out'):out=x;ref_out=ref_x
                    elif kind=='strided-out-last':
                        base=torch.full((*xshape,2),37.,dtype=dtype,device='npu:0');out=base[...,1]
                        ref_base=torch.full((*xshape,2),37.,dtype=dtype,device='npu:0');ref_out=ref_base[...,1]
                    else:out=torch.empty_like(x);ref_out=torch.empty_like(ref_x)
                    guards=storage_guards(dict(out=out))
                    expected=oracle.run(input=ref_x,other=ref_y,dim=dim,out=ref_out)
                    actual=candidate.run(input=x,other=y,dim=dim,out=out);torch.npu.synchronize()
                    torch.testing.assert_close(actual,expected,atol=tol['atol'],rtol=tol['rtol'],equal_nan=True)
                    assert torch._C._is_alias_of(actual,out),'返回未复用out'
                    if not kind.startswith('alias-out'):torch.testing.assert_close(x.cpu(),cpu_x,atol=0,rtol=0)
                    torch.testing.assert_close(y.cpu(),cpu_y,atol=0,rtol=0)
                    for _,storage,gap,saved in guards:assert torch.equal(storage[gap],saved),'out误写邻接存储'
                    row['passed']=True
                except Exception as error:row['error']=type(error).__name__+': '+str(error)[:2000]
                rows.append(row)
    if a.operator=='xlogy_scalar_tensor':
        parameters=[0.,-0.,2.,-3.,float('inf'),float('nan')]
        for dtype in [torch.float16,torch.bfloat16,torch.float32]:
            tolerance=next(c['tolerance'] for c in cases if c['inputs']['spec']['dtype']==str(dtype).split('.')[-1])
            for strided in [False,True]:
                cpu=torch.tensor([0.,-0.,-1.,1.,float('inf'),-float('inf'),float('nan'),2.,0.5],dtype=dtype).repeat(2,1)
                if strided:
                    base=torch.empty((*cpu.shape,2),dtype=dtype,device='npu:0');tensor=base[...,0]
                else:tensor=torch.empty_like(cpu,device='npu:0')
                for scalar in parameters:
                    row=dict(case='xlogy-scalars-'+str(dtype)+'-'+str(strided)+'-'+str(scalar),
                             check='runtime_scalar_and_special_value_sequence',passed=False)
                    try:
                        tensor.copy_(cpu);guards=storage_guards(dict(other=tensor));before=tensor.clone()
                        expected=oracle.run(self=scalar,other=tensor.clone());actual=candidate.run(self=scalar,other=tensor)
                        torch.npu.synchronize()
                        torch.testing.assert_close(actual,expected,atol=tolerance['atol'],rtol=tolerance['rtol'],equal_nan=True)
                        torch.testing.assert_close(tensor,before,atol=0,rtol=0,equal_nan=True)
                        assert not torch._C._is_alias_of(actual,tensor),'输出复用了输入'
                        for _,storage,gap,saved in guards:assert torch.equal(storage[gap],saved),'误写邻接存储'
                        row['passed']=True
                    except Exception as error:row['error']=type(error).__name__+': '+str(error)[:2000]
                    rows.append(row)
    if a.operator=='hardshrink_backward':
        for dtype in [torch.float16,torch.bfloat16,torch.float32]:
            for lambd in [0.,0.5,1.25]:
                row=dict(case='hardshrink-boundary-'+str(dtype)+'-'+str(lambd),check='threshold_neighbors_and_special_gradients',passed=False)
                try:
                    bound=torch.tensor([-lambd,lambd],dtype=dtype)
                    lower=torch.nextafter(bound,torch.full_like(bound,-float('inf')))
                    upper=torch.nextafter(bound,torch.full_like(bound,float('inf')))
                    cpu=torch.cat([torch.tensor([-float('inf'),float('inf'),float('nan'),0.,-0.],dtype=dtype),bound,lower,upper])
                    grad_cpu=torch.tensor([1.,-1.,2.,0.,-0.,3.,-3.,float('nan'),float('inf'),-float('inf'),7.],dtype=dtype)
                    xb=torch.empty((cpu.numel(),3),dtype=dtype,device='npu:0');xb[:,1]=cpu.to('npu:0');x=xb[:,1]
                    gb=torch.empty((cpu.numel(),2),dtype=dtype,device='npu:0');gb[:,0]=grad_cpu.to('npu:0');grad=gb[:,0]
                    inputs=dict(grad_out=grad,self_t=x,lambd=lambd);guards=storage_guards(inputs)
                    before={name:value.clone() for name,value in inputs.items() if isinstance(value,torch.Tensor)}
                    expected=oracle.run(grad_out=grad.clone(),self_t=x.clone(),lambd=lambd)
                    actual=candidate.run(**inputs);torch.npu.synchronize()
                    row.update(reference_device='npu:0',self_values=[str(v) for v in cpu.tolist()],
                        gradient_values=[str(v) for v in grad_cpu.tolist()],
                        npu_reference=[str(v) for v in expected.cpu().tolist()],
                        candidate=[str(v) for v in actual.cpu().tolist()])
                    try:
                        cpu_reference=oracle.run(grad_out=grad_cpu,self_t=cpu,lambd=lambd)
                        row['cpu_observation']=[str(v) for v in cpu_reference.tolist()]
                    except Exception as observation_error:
                        row['cpu_observation_error']=str(observation_error)
                    torch.testing.assert_close(actual,expected,atol=0,rtol=0,equal_nan=True)
                    for name,value in before.items():torch.testing.assert_close(inputs[name],value,atol=0,rtol=0,equal_nan=True)
                    for _,storage,gap,saved in guards:assert torch.equal(storage[gap],saved),'误写邻接存储'
                    assert not torch._C._is_alias_of(actual,x) and not torch._C._is_alias_of(actual,grad),'非原地结果复用了输入'
                    row['passed']=True
                except Exception as error:row['error']=type(error).__name__+': '+str(error)[:2000]
                rows.append(row)
    if a.operator=='expm1_':
        for dtype in [torch.float16,torch.bfloat16,torch.float32]:
            row=dict(case='expm1-small-'+str(dtype),check='small_magnitude_relative_accuracy',passed=False)
            try:
                cpu=torch.tensor([0.,-0.,1.e-8,-1.e-8,1.e-6,-1.e-6,1.e-4,-1.e-4,
                    0.03,-0.03,0.0624,-0.0624,0.0625,-0.0625,float('nan'),float('inf'),-float('inf')],dtype=dtype)
                base=torch.empty((cpu.numel(),2),dtype=dtype,device='npu:0');base[:,0]=cpu.to('npu:0');tensor=base[:,0]
                guards=storage_guards(dict(A=tensor));expected=oracle.run(A=tensor.clone());actual=candidate.run(A=tensor)
                torch.npu.synchronize();tolerance=next(c['tolerance'] for c in cases if c['inputs']['spec']['dtype']==str(dtype).split('.')[-1])
                cpu_reference=oracle.run(A=cpu.clone())
                observed=actual.cpu();target=expected.cpu()
                row.update(input_values=[str(v) for v in cpu.tolist()],npu_reference=[str(v) for v in target.tolist()],
                    candidate=[str(v) for v in observed.tolist()],cpu_reference=[str(v) for v in cpu_reference.tolist()],
                    comparison_policy='NPU一致性沿用正式合同容差；额外小值稳定性对照CPU PyTorch，不以有消去误差的NPU结果作零绝对误差真值')
                torch.testing.assert_close(actual,expected,atol=tolerance['atol'],rtol=tolerance['rtol'],equal_nan=True)
                row['npu_contract_passed']=True
                torch.testing.assert_close(observed,cpu_reference,atol=0,rtol=tolerance['rtol'],equal_nan=True)
                row['cpu_small_relative_passed']=True
                zero=cpu_reference==0
                row['npu_reference_zero_signbits']=torch.signbit(target[zero]).tolist()
                row['cpu_reference_zero_signbits']=torch.signbit(cpu_reference[zero]).tolist()
                row['candidate_zero_signbits']=torch.signbit(observed[zero]).tolist()
                assert row['candidate_zero_signbits']==row['cpu_reference_zero_signbits'],'expm1零符号不符合CPU数学参考'
                assert torch._C._is_alias_of(actual,tensor),'原地返回别名错误'
                for _,storage,gap,saved in guards:assert torch.equal(storage[gap],saved),'误写邻接存储'
                row['passed']=True
            except Exception as error:row['error']=type(error).__name__+': '+str(error)[:2000]
            rows.append(row)
    if a.operator in {'round_out','special_round_out'}:
        for dtype in [torch.float16,torch.bfloat16,torch.float32]:
            tolerance=next(c['tolerance'] for c in cases if c['inputs']['spec']['dtype']==str(dtype).split('.')[-1])
            for decimals in [-1,0,1,2]:
                row=dict(case='round-edges-'+str(dtype)+'-'+str(decimals),check='rounding_ties_and_layout',passed=False)
                try:
                    cpu=torch.tensor([0.,-0.,0.5,-0.5,1.5,-1.5,2.5,-2.5,15.,-15.,25.,-25.,
                        0.125,-0.125,float('nan'),float('inf'),-float('inf'),2048.,-2048.],dtype=dtype)
                    base=torch.empty((cpu.numel(),2),dtype=dtype,device='npu:0');base[:,0]=cpu.to('npu:0');tensor=base[:,0]
                    out_base=torch.empty((cpu.numel(),3),dtype=dtype,device='npu:0');target=out_base[:,1]
                    name='input' if a.operator=='round_out' else 'self'
                    inputs={name:tensor,'decimals':decimals,'out':target}
                    guards=storage_guards(inputs);before=tensor.clone()
                    expected=oracle.run(**{name:tensor.clone(),'decimals':decimals,'out':torch.empty_like(tensor)})
                    actual=candidate.run(**inputs);torch.npu.synchronize()
                    assert torch._C._is_alias_of(actual,target),'out返回别名错误'
                    torch.testing.assert_close(actual,expected,atol=0 if decimals==0 else tolerance['atol'],
                                               rtol=0 if decimals==0 else tolerance['rtol'],equal_nan=True)
                    torch.testing.assert_close(tensor,before,atol=0,rtol=0,equal_nan=True)
                    zero=expected==0
                    assert torch.equal(torch.signbit(actual[zero]),torch.signbit(expected[zero])),'舍入零符号不同'
                    for _,storage,gap,saved in guards:assert torch.equal(storage[gap],saved),'误写邻接存储'
                    row['passed']=True
                except Exception as error:row['error']=type(error).__name__+': '+str(error)[:2000]
                rows.append(row)
    if a.operator=='native_group_norm':
        for dtype in ['float16','bfloat16','float32']:
            case=next(c for c in cases if c['inputs']['spec']['dtype']==dtype and c['inputs']['spec'].get('affine'))
            for absent in ['weight','bias']:
                row=dict(case='groupnorm-optional-'+dtype+'-without-'+absent,check='independent_optional_affine',passed=False)
                try:
                    ref_inputs=oracle.gen_inputs(case,'npu:0');inputs=oracle.gen_inputs(case,'npu:0')
                    ref_inputs[absent]=None;inputs[absent]=None
                    before={name:value.clone() for name,value in inputs.items() if isinstance(value,torch.Tensor)}
                    expected=oracle.run(**ref_inputs);actual=candidate.run(**inputs);torch.npu.synchronize()
                    assert len(actual)==len(expected)==3
                    for ref,out in zip(expected,actual):
                        torch.testing.assert_close(out,ref,atol=case['tolerance']['atol'],rtol=case['tolerance']['rtol'],equal_nan=True)
                    for name,value in before.items():
                        torch.testing.assert_close(inputs[name],value,atol=0,rtol=0,equal_nan=True)
                    row['passed']=True
                except Exception as error:row['error']=type(error).__name__+': '+str(error)[:2000]
                rows.append(row)
    if a.operator=='fix':
        for dtype in [torch.float16,torch.bfloat16,torch.float32]:
            row=dict(case='fix-edges-'+str(dtype),check='target_truncation_edges',passed=False)
            try:
                # 覆盖正负零、整数两侧、非有限值、可精确表示大整数及偏移视图。
                cpu=torch.tensor([0.,-0.,0.25,-0.25,0.5,-0.5,1.,-1.,1.75,-1.75,
                    2.,-2.,float('nan'),float('inf'),-float('inf'),65504.,-65504.],dtype=dtype)
                around=torch.tensor([1.,-1.,2.,-2.],dtype=dtype)
                cpu=torch.cat([cpu,torch.nextafter(around,torch.full_like(around,float('inf'))),
                               torch.nextafter(around,torch.full_like(around,-float('inf')))])
                base=torch.empty((cpu.numel(),2),dtype=dtype,device='npu:0');base[:,0]=cpu.to('npu:0')
                tensor=base[:,0];before=tensor.clone();guards=storage_guards(dict(self=tensor))
                expected=oracle.run(self=tensor.clone());actual=candidate.run(self=tensor);torch.npu.synchronize()
                torch.testing.assert_close(actual,expected,atol=0,rtol=0,equal_nan=True)
                torch.testing.assert_close(tensor,before,atol=0,rtol=0,equal_nan=True)
                assert not torch._C._is_alias_of(actual,tensor),'非原地算子返回了输入别名'
                zero=expected==0
                row['npu_reference_zero_signbits']=torch.signbit(expected[zero]).tolist()
                row['candidate_zero_signbits']=torch.signbit(actual[zero]).tolist()
                assert row['candidate_zero_signbits']==row['npu_reference_zero_signbits'],'截断零符号不同'
                for _,storage,gap,saved in guards:assert torch.equal(storage[gap],saved),'误写邻接存储'
                row['passed']=True
            except Exception as error:row['error']=type(error).__name__+': '+str(error)[:2000]
            rows.append(row)
    if a.operator=='nan_to_num_':
        parameter_sets=[('defaults',dict(nan=0.,posinf=None,neginf=None)),
                        ('finite',dict(nan=2.5,posinf=3.,neginf=-4.)),
                        ('nonfinite',dict(nan=float('inf'),posinf=0.,neginf=-0.))]
        for dtype in [torch.float16,torch.bfloat16,torch.float32]:
            for strided in [False,True]:
                cpu=torch.tensor([0.,-0.,float('nan'),float('inf'),-float('inf'),-2.,3.],dtype=dtype).repeat(2,1)
                if strided:
                    base=torch.full((2,7,2),37.,dtype=dtype,device='npu:0');tensor=base[...,0]
                else:tensor=torch.empty_like(cpu,device='npu:0')
                for name,parameters in parameter_sets:
                    row=dict(case='nan-parameters-'+str(dtype)+'-'+str(strided)+'-'+name,
                             check='nan_to_num_parameters',parameter_sequence_reuses_storage=True,passed=False)
                    try:
                        tensor.copy_(cpu)
                        expected_input=tensor.clone();guards=storage_guards(dict(A=tensor))
                        expected=oracle.run(A=expected_input,**parameters);actual=candidate.run(A=tensor,**parameters)
                        torch.npu.synchronize()
                        assert torch._C._is_alias_of(actual,tensor),'原地返回别名错误'
                        torch.testing.assert_close(actual,expected,atol=0,rtol=0,equal_nan=True)
                        zero=expected==0
                        assert torch.equal(torch.signbit(actual[zero]),torch.signbit(expected[zero])),'零符号与目标参考不同'
                        for _,storage,gap,before in guards:assert torch.equal(storage[gap],before),'误写非目标存储'
                        row['passed']=True
                    except Exception as error:row['error']=type(error).__name__+': '+str(error)[:2000]
                    rows.append(row)
    if a.operator in {'negative_','expm1_'}:
        # 补足原合同之外的空输入、偏移、转置及非线性stride，不改变正式容差。
        for dtype in [torch.float16,torch.bfloat16,torch.float32]:
            generator=torch.Generator().manual_seed(91731)
            original=torch.randn((9,13),generator=generator).to(dtype=dtype,device='npu:0')
            makers={
                'empty':lambda:torch.empty(0,dtype=dtype,device='npu:0'),
                'scalar':lambda:original[0,0].clone(),
                'offset':lambda:original.flatten()[1:67],
                'transpose':lambda:original.t(),
                'slice2d':lambda:original[1:8:2,2:12:3],
                'permuted3d':lambda:original.flatten()[:72].reshape(3,4,6).permute(2,0,1),
                'tail':lambda:original.flatten()[:113],
            }
            for kind,maker in makers.items():
                row=dict(case='additional-'+str(dtype)+'-'+kind,passed=False,check='independent_layout_semantics')
                try:
                    # 每次生成独立存储，保持视图的stride和storage_offset。
                    original=torch.randn((9,13),generator=torch.Generator().manual_seed(91731)).to(dtype=dtype,device='npu:0')
                    actual_input=maker();reference=actual_input.clone()
                    expected=oracle.run(A=reference);before_guards=storage_guards(dict(A=actual_input))
                    actual=candidate.run(A=actual_input);torch.npu.synchronize()
                    assert torch._C._is_alias_of(actual,actual_input),'原地返回别名错误'
                    tol=next(case['tolerance'] for case in cases if case['inputs']['spec']['dtype']==str(dtype).split('.')[-1])
                    torch.testing.assert_close(actual,expected,atol=tol['atol'],rtol=tol['rtol'],equal_nan=True)
                    for name,storage,gap,before in before_guards:assert torch.equal(storage[gap],before),'非目标存储被修改'
                    row['passed']=True
                except Exception as error:row['error']=type(error).__name__+': '+str(error)[:2000]
                rows.append(row)
            row=dict(case='additional-'+str(dtype)+'-alignment-sequence',passed=False,check='same_shape_pointer_alignment')
            try:
                cpu=torch.randn((1024,),generator=torch.Generator().manual_seed(110730)).to(dtype)
                base=torch.full((1032,),37.,dtype=dtype,device='npu:0')
                tol=next(case['tolerance'] for case in cases if case['inputs']['spec']['dtype']==str(dtype).split('.')[-1])
                offsets=[0,1,3,0]
                for offset in offsets:
                    base.fill_(37.);tensor=base[offset:offset+1024];tensor.copy_(cpu)
                    expected=oracle.run(A=tensor.clone());actual=candidate.run(A=tensor);torch.npu.synchronize()
                    torch.testing.assert_close(actual,expected,atol=tol['atol'],rtol=tol['rtol'],equal_nan=True)
                    assert torch._C._is_alias_of(actual,tensor),'原地返回别名错误'
                    assert torch.all(base[:offset]==37).item() and torch.all(base[offset+1024:]==37).item(),'偏移视图误写邻接存储'
                row.update(passed=True,offset_sequence=offsets,elements=1024)
            except Exception as error:row['error']=type(error).__name__+': '+str(error)[:2000]
            rows.append(row)
            if a.operator=='negative_':
                row=dict(case='additional-'+str(dtype)+'-signed-zero',passed=False,check='signed_zero_and_extremes')
                try:
                    cpu=torch.tensor([0.,-0.,float('nan'),float('inf'),-float('inf'),1.,-1.],dtype=dtype)
                    tensor=cpu.to('npu:0');reference=cpu.to('npu:0')
                    expected=oracle.run(A=reference);actual=candidate.run(A=tensor);torch.npu.synchronize();actual=actual.cpu();expected=expected.cpu()
                    row['target_reference']='当前NPU上的PyTorch oracle'
                    row['npu_reference_signbits']=torch.signbit(expected[:2]).tolist()
                    row['cpu_ieee_signbits']=torch.signbit((-cpu)[:2]).tolist()
                    row['candidate_signbits']=torch.signbit(actual[:2]).tolist()
                    row['cpu_ieee_matches_target']=row['npu_reference_signbits']==row['cpu_ieee_signbits']
                    torch.testing.assert_close(actual,expected,atol=0,rtol=0,equal_nan=True)
                    zero=expected==0
                    assert torch.equal(torch.signbit(actual[zero]),torch.signbit(expected[zero])),'负号未匹配本机PyTorch的正负零语义'
                    row['passed']=True
                except Exception as error:row['error']=type(error).__name__+': '+str(error)[:2000]
                rows.append(row)
    random_details=None
    if a.operator=='uniform':
        # 复用历史真机门禁，补齐范围/布局/stream及seed、offset高位与状态重放。
        details=a.output.with_name(a.output.stem+'-random-details.json')
        row=dict(case='uniform-extended-checker',check='random_checker',passed=False)
        try:
            checker_path=a.uniform_checker
            checker=imported('phase_uniform_checker',checker_path)
            if not details.exists():checker.check('uniform',a.source,details)
            legacy=json.loads(details.read_text(encoding='utf8'))
            assert legacy['source_sha256']==hashlib.sha256(a.source.read_bytes()).hexdigest(),'随机门禁源码不同'
            assert legacy['checker_sha256']==hashlib.sha256(checker_path.read_bytes()).hexdigest(),'随机门禁版本不同'
            random_details=dict(path=str(details),sha256=hashlib.sha256(details.read_bytes()).hexdigest(),
                                checker_sha256=legacy['checker_sha256'])
            for item in legacy['cases']:
                rows.append(dict(case='uniform-layout-'+item['dtype']+'-'+str(item['case']),
                    check='uniform_layout_range_stream',passed=item['status']=='passed',error=item.get('error')))
            for item in legacy['generator_semantics']['cases']:
                rows.append(dict(case='uniform-rng-'+item['case'],check='uniform_generator_state',
                    passed=item['status']=='passed',error=item.get('error')))
            row['passed']=legacy['status']=='passed'
        except Exception as error:row['error']=type(error).__name__+': '+str(error)[:2000]
        rows.append(row)
    checker_sha=hashlib.sha256(CHECKER_BYTES).hexdigest()
    checker_source=a.output.with_name('validator-'+checker_sha+'.py')
    if checker_source.exists():assert checker_source.read_bytes()==CHECKER_BYTES,'检查器快照身份不同'
    else:checker_source.write_bytes(CHECKER_BYTES)
    result=dict(status='passed' if all(r['passed'] for r in rows) else 'failed',operator=a.operator,
                suite='hgkda-next25-supplement-v2',checker_sha256=checker_sha,
                checker_source=dict(path=str(checker_source),sha256=checker_sha),
                reproduce_command=[sys.executable,str(checker_source),'--operator',a.operator,'--leaf',str(a.leaf),
                    '--source',str(a.source),'--output',str(a.output.with_name('reproduced-'+a.output.name)),
                    '--uniform-checker',str(a.uniform_checker)],
                target_reference='当前NPU上的PyTorch oracle；CPU IEEE仅作旁列观察',
                source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),passed=sum(r['passed'] for r in rows),total=len(rows),cases=rows)
    if random_details:result['random_details']=random_details
    if a.operator=='expm1_':result['target_reference']='正式一致性沿用NPU合同容差；小值相对精度和零符号额外对照CPU PyTorch，原KGS合同不改'
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False))
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
