
from pathlib import Path
import torch,json,hashlib
torch.set_num_threads(4)
rows=[]
def model(inp,index,src):
    width=inp.shape[-1];B=1<<(width-1).bit_length();LOG=B.bit_length()-1
    lane=torch.arange(B)[None,:].expand(inp.shape[0],-1)
    padded=torch.full(lane.shape,B,dtype=torch.int64);padded[:,:width]=index
    keys=(padded*B+lane).float().sort(-1).values
    keyi=keys.to(torch.int64);target=keyi>>LOG
    values=torch.zeros(lane.shape);values[:,:width]=src.float()
    values=values.gather(1,(keyi&(B-1)).clamp(max=width-1))
    values=torch.where(target<width,values,0.)
    heads=(lane==0)|(target!=torch.roll(target,1,-1))
    delta=1
    while delta<B:
        right_s=values[:,delta:];right_h=heads[:,delta:]
        new_s=torch.where(right_h,right_s,values[:,:-delta]+right_s)
        new_h=heads[:,:-delta]|right_h
        values=torch.cat([values[:,:delta],new_s],-1)
        heads=torch.cat([heads[:,:delta],new_h],-1)
        delta*=2
    destinations=torch.arange(width)[None,:].expand(inp.shape[0],-1)
    lo=torch.searchsorted(keys.contiguous(),(destinations*B).float().contiguous())
    hi=torch.searchsorted(keys.contiguous(),((destinations+1)*B).float().contiguous())
    sums=values.gather(1,(hi-1).clamp(min=0,max=B-1))
    sums=torch.where(hi>lo,sums,0.)
    return (inp.float()+sums).to(inp.dtype)
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 for width in [257,1024,4096]:
  for pattern in ['random-1','random-2','positive-large','negative-large','empty-target','alternating-targets']:
   g=torch.Generator().manual_seed(9138+int(pattern=='random-2'))
   shape=(5,width)
   inp=torch.randn(shape,generator=g).to(dtype);src=torch.randn(shape,generator=g).to(dtype)
   index=torch.randint(width,shape,generator=g)
   if pattern.endswith('large'):
    inp.zero_();src.fill_((1 if pattern=='positive-large' else -1)*(65504 if dtype==torch.float16 else 1e20))
    index.zero_();src[:,-1]=.25;index[:,-1]=1
   elif pattern=='empty-target':index.fill_(width//2)
   elif pattern=='alternating-targets':index[:]=torch.arange(width)%2
   actual=model(inp,index,src);expected=torch.scatter_add(inp,1,index,src)
   tol=1e-4 if dtype==torch.float32 else .015
   ok=bool(torch.isclose(actual,expected,atol=tol,rtol=tol,equal_nan=True).all())
   rows.append(dict(dtype=str(dtype),width=width,pattern=pattern,passed=ok))
report=dict(scope='CPU数学模型，显式Hillis-Steele分段扫描；不是NPU编译或性能证明',
            total=len(rows),passed=sum(r['passed'] for r in rows),cases=rows)
path=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/scatter-segmented-20261023/knowledge/scatter-segmented-cpu.json')
raw=(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode()
if path.exists():assert path.read_bytes()==raw
else:path.write_bytes(raw)
print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False))
assert report['passed']==report['total'],[r for r in rows if not r['passed']]
