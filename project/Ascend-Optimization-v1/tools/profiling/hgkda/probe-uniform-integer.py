#!/usr/bin/env python3
"""有限整数原语见证：区别乘法、逻辑移位和位运算故障，不计优化成绩。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    import torch
    import torch_npu
    import triton
    import triton.language as tl
    globals().update(tl=tl,triton=triton)
    assert torch.npu.device_count()==1

    @triton.jit
    def compute(A,B,O,MODE:tl.constexpr,BLOCK:tl.constexpr):
        i=tl.arange(0,BLOCK)
        x=tl.load(A+i).to(tl.uint32,bitcast=True)
        y=tl.load(B+i).to(tl.uint32,bitcast=True)
        if MODE==0:v=x*y
        elif MODE==1:v=(x&65535)*(y&65535)
        elif MODE==2:v=x>>16
        elif MODE==3:v=x+y
        elif MODE==4:v=x^y
        elif MODE==5:v=x&y
        elif MODE==6:v=x|y
        elif MODE==7:v=x<<16
        elif MODE==8:v=tl.umulhi(x,y)
        else:
            x0=x&65535;x1=x>>16;y0=y&65535;y1=y>>16
            u=x0*y0;t=x1*y0+(u>>16);w=x0*y1+(t&65535)
            v=x1*y1+(t>>16)+(w>>16)
        tl.store(O+i,v.to(tl.int32,bitcast=True))

    mask=2**32-1
    rng=random.Random(19023)
    x=[0,1,65535,65536,2**31-1,2**31,mask,0x12345678]+[rng.getrandbits(32) for _ in range(56)]
    y=[0xD2511F53 if i%2==0 else 0xCD9E8D57 for i in range(64)]
    signed=lambda v:v if v<2**31 else v-2**32
    xx=torch.tensor([signed(v) for v in x],dtype=torch.int32,device='npu:0')
    yy=torch.tensor([signed(v) for v in y],dtype=torch.int32,device='npu:0')
    out=torch.empty_like(xx)
    functions=[('mul32',lambda v,w:v*w),('mul16',lambda v,w:(v&65535)*(w&65535)),
               ('shr16',lambda v,w:v>>16),('add32',lambda v,w:v+w),('xor',lambda v,w:v^w),
               ('and',lambda v,w:v&w),('or',lambda v,w:v|w),('shl16',lambda v,w:v<<16),
               ('umulhi',lambda v,w:v*w>>32),('split16_hi',lambda v,w:v*w>>32)]
    rows=[]
    for mode,(name,fn) in enumerate(functions):
        row=dict(operation=name,status='failed')
        try:
            compiled=compute[(1,)](xx,yy,out,MODE=mode,BLOCK=64)
            torch.npu.synchronize()
            got=[int(v)&mask for v in out.cpu().tolist()]
            expected=[fn(v,w)&mask for v,w in zip(x,y)]
            failures=[dict(index=i,a=x[i],b=y[i],actual=v,expected=w) for i,(v,w) in enumerate(zip(got,expected)) if v!=w]
            row.update(mismatches=len(failures),failures=failures[:8],status='passed' if not failures else 'failed')
            if failures:
                directory=a.output.parent/(a.output.stem+'-ir');directory.mkdir(parents=True,exist_ok=True)
                row['ir']=[]
                for kind,content in getattr(compiled,'asm',{}).items():
                    if not isinstance(content,(str,bytes)):continue
                    raw=content.encode() if isinstance(content,str) else content
                    path=directory/(name+'.'+kind);path.write_bytes(raw)
                    row['ir'].append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),kind=kind))
        except Exception as error:row['error']=type(error).__name__+': '+str(error)
        rows.append(row)
    result=dict(status='completed',cases=rows,inputs=dict(a=x,b=y),checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='64组确定性整数输入，独立CPU精确整数参考；非性能评测，只界定固定后端上的有限原语反例。')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='completed',operations=[{k:v for k,v in row.items() if k not in ['ir','failures']} for row in rows]),ensure_ascii=False))


if __name__=='__main__':main()
