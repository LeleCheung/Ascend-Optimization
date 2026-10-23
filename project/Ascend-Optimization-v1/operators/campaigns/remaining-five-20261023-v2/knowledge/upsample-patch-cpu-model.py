"""CPU验证连续输入行块+halo的2倍双线性插值；不代表NPU编译或性能。"""
import hashlib
import json
from pathlib import Path
import torch


def model(x, r):
    n, c, ih, iw = x.shape
    oh, ow = 2*ih, 2*iw
    out = torch.empty((n,c,oh,ow), dtype=x.dtype)
    rp = 1 << (r+1).bit_length()
    wp = 1 << (iw-1).bit_length()
    col = torch.arange(iw)
    for ni in range(n):
        for ci in range(c):
            for q in range(0,ih,r):
                # 一次连续GM读取的CPU等价模型，按通道边界屏蔽halo。
                first=q-1
                patch=torch.zeros((rp,wp),dtype=torch.float32)
                lo=max(first,0);hi=min(first+rp,ih)
                patch[lo-first:hi-first,:iw]=x[ni,ci,lo:hi,:].float()
                yy=torch.arange(2*q,min(2*(q+r),oh))
                pos=torch.clamp((yy.float()+.5)*.5-.5,min=0)
                a=pos.to(torch.int64).clamp(max=ih-1)
                b=(a+1).clamp(max=ih-1)
                w=(pos-a.float())[:,None]
                v=(1-w)*patch[a-first,:iw]+w*patch[b-first,:iw]
                left=v[:,(col-1).clamp(min=0)]
                right=v[:,(col+1).clamp(max=iw-1)]
                out[ni,ci,yy[:,None],2*col]=(.25*left+.75*v).to(x.dtype)
                out[ni,ci,yy[:,None],2*col+1]=(.75*v+.25*right).to(x.dtype)
    return out


if __name__=='__main__':
    torch.set_num_threads(2)
    rows=[]
    shapes=[(1,1,1,1),(1,2,2,3),(2,3,5,9),(1,3,7,17),
            (1,2,8,8),(1,2,9,9),(1,2,16,32),(1,1,17,33)]
    for dtype in [torch.float16,torch.bfloat16,torch.float32]:
        for shape in shapes:
            x=torch.randn(shape,generator=torch.Generator().manual_seed(20261023)).to(dtype)
            want=torch.ops.aten.upsample_bilinear2d(x,[2*shape[2],2*shape[3]],False)
            for r in [2,4,8]:
                row=dict(dtype=str(dtype),shape=shape,r=r,status='failed')
                try:
                    got=model(x,r)
                    tol=1e-5 if dtype==torch.float32 else .015
                    torch.testing.assert_close(got,want,atol=tol,rtol=tol)
                    row.update(status='passed',max_abs=(got.float()-want.float()).abs().max().item())
                except Exception as e:row['error']=type(e).__name__+': '+str(e)
                rows.append(row)
    result=dict(scope='CPU索引/halo/插值等价模型，不是NPU编译、数值或性能证明',
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                total=len(rows),passed=sum(row['status']=='passed' for row in rows),cases=rows)
    output=Path(__file__).with_name('upsample-patch-cpu-result.json')
    assert not output.exists(),output
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(total=result['total'],passed=result['passed'],failures=[r for r in rows if r['status']!='passed'])))
