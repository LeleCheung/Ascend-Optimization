from pathlib import Path
import importlib.util,json,hashlib,torch,torch_npu,triton
import triton.language as tl
root=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1')
batch=root/'operators/campaigns/remaining-five-20261023-v2'
source=batch/'knowledge/fractional-bitround-intervals.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='4228f95c27e2dedb1edad3d8c37441a96e2c6384c84d423533be7413b30a0a66'
spec=importlib.util.spec_from_file_location('bitround_candidate',source)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rounder=m._round_coordinate
@triton.jit
def rounding(X,A,B,N:tl.constexpr,DT:tl.constexpr,BLOCK:tl.constexpr):
 lane=tl.arange(0,BLOCK);v=tl.load(X+lane,mask=lane<N,other=0.)
 tl.store(A+lane,v.to(DT).to(tl.float32),mask=lane<N)
 tl.store(B+lane,rounder(v,DT),mask=lane<N)
values=torch.tensor([0.,2**-25,3*2**-25,2**-24,2**-14,.499,.4995,.4999,.5,1.499,10.499,14.499,14.4995,14.4999,14.5,65504.,65520.]+[float(i)/31 for i in range(31)],dtype=torch.float32)
x=values.to('npu:0');rows=[]
for dtype,dt in [(torch.float16,tl.float16),(torch.bfloat16,tl.bfloat16)]:
 a=torch.empty_like(x);b=torch.empty_like(x)
 rounding[(1,)](x,a,b,values.numel(),dt,triton.next_power_of_2(values.numel()))
 torch.npu.synchronize();expected=values.to(dtype).float()
 rows.append(dict(dtype=str(dtype),intrinsic_equal=torch.equal(a.cpu(),expected),bitround_equal=torch.equal(b.cpu(),expected),
                  mismatches=[dict(input=float(values[i]),expected=float(expected[i]),intrinsic=float(a.cpu()[i]),bitround=float(b.cpu()[i])) for i in range(values.numel()) if a.cpu()[i]!=expected[i] or b.cpu()[i]!=expected[i]]))
 print('ROUNDING',json.dumps(rows[-1]),flush=True)
out=batch/'research/fractional_max_pool3d/bitround-interval-probe'
(out/'rounding-components.json').write_text(json.dumps(dict(cases=rows),indent=2)+'\n')
assert all(r['bitround_equal'] for r in rows),'位运算舍入模型未匹配CPU，不继续完整检查'
checks=[]
for filename,name in [('check-remaining-five.py','semantics.json'),('check-remaining-special.py','special.json')]:
 spec=importlib.util.spec_from_file_location('bitround_'+name.replace('.','_'),root/'tools/profiling/hgkda'/filename)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 code=module.check('fractional_max_pool3d',source,out/name)
 data=json.loads((out/name).read_text())
 print('CHECK',json.dumps(dict(name=name,status=data['status'],passed=data['passed'],total=data['total'],failures=[r for r in data['cases'] if r['status']!='passed']),ensure_ascii=False),flush=True)
 checks.append(code)
raise SystemExit(int(any(checks)))
