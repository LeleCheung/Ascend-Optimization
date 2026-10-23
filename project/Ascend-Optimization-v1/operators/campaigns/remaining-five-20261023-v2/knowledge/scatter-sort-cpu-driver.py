import subprocess,json
script=r'''
from pathlib import Path
import json,torch,hashlib
torch.set_num_threads(4)
rows=[]
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 for n in [17,128,1024,2048]:
  for seed in [9721,9722]:
   g=torch.Generator().manual_seed(seed)
   inp=torch.randn((9,n),generator=g).to(dtype)
   src=torch.randn((9,n),generator=g).to(dtype)
   index=torch.randint(n,(9,n),generator=g)
   key=index*n+torch.arange(n)
   sorted_key=key.sort(dim=-1).values
   assert sorted_key.max()<2**24
   assert torch.equal(sorted_key,sorted_key.float().to(torch.int64))
   lane=sorted_key%n
   values=src.float().gather(1,lane)
   prefix=values.cumsum(-1)
   targets=torch.arange(n)[None,:].expand(9,-1)
   lo=torch.searchsorted(sorted_key.contiguous(),(targets*n).contiguous(),right=False)
   hi=torch.searchsorted(sorted_key.contiguous(),((targets+1)*n).contiguous(),right=False)
   a=prefix.gather(1,(hi-1).clamp(min=0))
   b=prefix.gather(1,(lo-1).clamp(min=0))
   sums=torch.where(hi>0,a,0)-torch.where(lo>0,b,0)
   out=(inp.float()+sums).to(dtype)
   expected=torch.scatter_add(inp,1,index,src)
   atol=1e-4 if dtype==torch.float32 else .015
   passed=torch.isclose(out,expected,atol=atol,rtol=atol).all().item()
   rows.append(dict(dtype=str(dtype),width=n,seed=seed,passed=passed,max_abs_error=(out.float()-expected.float()).abs().max().item()))
report=dict(scope='独立CPU数学模型；不是NPU编译、执行或性能证明',cases=rows,total=len(rows),passed=sum(r['passed'] for r in rows),
 method='FP32精确编码目标index和原位置排序；按排序原位置读取src并FP32前缀和，目标区间二分查找后相减；加inp并转回dtype')
path=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/remaining-five-20261023/knowledge/scatter-sort-cpu.json')
assert not path.exists();path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False))
'''
result=subprocess.run(['docker','exec','-i','tle_yy','/usr/local/python3.11.15/bin/python3.11','-'],input=script.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
assert result.returncode==0,result.stderr.decode(errors='replace')
print(result.stdout.decode())
