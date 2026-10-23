from pathlib import Path
import importlib.util,json,hashlib,torch,torch_npu,triton
import triton.language as tl
from triton.language.extra.cann import extension as al
assert torch.npu.device_count()==1

@triton.jit
def combine(lv,lh,rv,rh):
    return tl.where(rh,rv,lv+rv),lh|rh

@triton.jit
def components(X,I,V,S,H,Q,A,B:tl.constexpr,LOG:tl.constexpr):
    lane=tl.arange(0,B)
    values=tl.load(X+lane)
    target=tl.load(I+lane)
    previous=tl.gather(target.to(tl.float32),tl.maximum(lane-1,0),axis=0).to(tl.int32)
    heads=(lane==0)|(target!=previous)
    sums,flags=tl.associative_scan((values,heads),axis=0,combine_fn=combine)
    manual=values
    for shift in tl.static_range(LOG):
        partner=tl.maximum(lane-(1<<shift),0)
        prev_target=tl.gather(target.to(tl.float32),partner,axis=0).to(tl.int32)
        prev_sum=tl.gather(manual,partner,axis=0)
        manual=manual+tl.where((lane>=(1<<shift))&(target==prev_target),prev_sum,0.)
    tl.store(V+lane,values)
    tl.store(S+lane,sums)
    tl.store(H+lane,heads.to(tl.int32))
    tl.store(Q+lane,flags.to(tl.int32))
    tl.store(A+lane,manual)

rows=[]
for size in [32,512]:
 x=torch.zeros(size,device='npu:0');x[0]=2**27;x[-1]=.25
 ids=torch.zeros(size,device='npu:0',dtype=torch.int32);ids[-1]=1
 v=torch.empty_like(x);s=torch.empty_like(x);h=torch.empty_like(ids);q=torch.empty_like(ids);a=torch.empty_like(x)
 components[(1,)](x,ids,v,s,h,q,a,size,size.bit_length()-1)
 torch.npu.synchronize()
 rows.append(dict(size=size,values=v.cpu().tolist(),heads=h.cpu().tolist(),flags=q.cpu().tolist(),
                  tuple_scan=s.cpu().tolist(),manual_scan=a.cpu().tolist()))
 print('SCAN_COMPONENT',json.dumps(rows[-1]),flush=True)
root=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1')
source=root/'operators/campaigns/scatter-segmented-20261023/knowledge/scatter-manual-segmented-v2-prototype.py'
spec=importlib.util.spec_from_file_location('manual_segmented_checker',root/'tools/profiling/hgkda/check-scatter-dynamic-range.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
out=root/'operators/campaigns/remaining-five-20261023-v2/research/scatter_add/manual-segmented-v2-probe'
(out/'components.json').write_text(json.dumps(dict(cases=rows),indent=2)+'\n')
code=module.check(source,out/'dynamic-range.json')
print('CHECK_RESULT',json.dumps(json.loads((out/'dynamic-range.json').read_text()),ensure_ascii=False),flush=True)
raise SystemExit(code)
