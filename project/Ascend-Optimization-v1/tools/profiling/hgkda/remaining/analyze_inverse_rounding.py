"""离线FP32舍入实验，比较LU倒数缩放/FMA；不作为设备正确性或性能成绩。"""
import itertools
import json
import math
from pathlib import Path
import struct

PROJECT=Path(__file__).resolve().parents[4]
BATCH=PROJECT/'operators/campaigns/remaining-nine-20261030'


def f32(x):return struct.unpack('f',struct.pack('f',x))[0]


def inverse(matrix,reciprocal,fused,solve_fused):
    n=len(matrix);a=[[f32(x) for x in row] for row in matrix];pivots=[]
    madd=lambda x,y,z:f32(math.fma(x,y,z)) if fused else f32(f32(x*y)+z)
    solve=lambda x,y,z:f32(math.fma(x,y,z)) if solve_fused else f32(f32(x*y)+z)
    for j in range(n):
        p=max(range(j,n),key=lambda k:abs(a[k][j]));pivots.append(p);a[j],a[p]=a[p],a[j]
        pivot=a[j][j]
        if pivot==0:return dict(info=j+1)
        r=f32(1.0/pivot)
        for i in range(j+1,n):a[i][j]=f32(a[i][j]*r) if reciprocal else f32(a[i][j]/pivot)
        for i in range(j+1,n):
            for k in range(j+1,n):a[i][k]=madd(-a[i][j],a[j][k],a[i][k])
    rhs=[[float(i==j) for j in range(n)] for i in range(n)]
    for j,p in enumerate(pivots):rhs[j],rhs[p]=rhs[p],rhs[j]
    for j in range(n):
        for i in range(j+1,n):
            for k in range(n):rhs[i][k]=solve(-a[i][j],rhs[j][k],rhs[i][k])
    for j in range(n-1,-1,-1):
        for k in range(n):rhs[j][k]=f32(rhs[j][k]/a[j][j])
        for i in range(j):
            for k in range(n):rhs[i][k]=solve(-a[i][j],rhs[j][k],rhs[i][k])
    return dict(info=0,inverse=rhs,lu=a)


def main():
    if not hasattr(math,'fma'):raise RuntimeError('离线脚本需Python>=3.13的math.fma；不在设备上运行')
    probe=json.loads((BATCH.parent/'next25-20261029-v2/analysis/linalg_inv_ex/contract-probe-fc75b4022866.json').read_text(encoding='utf8'))
    case=probe['cases'][1];matrix=case['input']['values'];ref=case['outputs'][0]['npu_reference']['values']
    rows=[]
    for reciprocal,fused,solve_fused in itertools.product([False,True],repeat=3):
        value=inverse(matrix,reciprocal,fused,solve_fused)
        if value['info']==0:
            errors=[abs(value['inverse'][i][j]-ref[i][j]) for i in range(3) for j in range(3)]
            matched=all(abs(value['inverse'][i][j]-ref[i][j])<=1.e-4+1.e-4*abs(ref[i][j]) for i in range(3) for j in range(3))
            value.update(max_absolute_difference=max(errors),matched=matched)
        rows.append(dict(reciprocal_scale=reciprocal,lu_fused=fused,solve_fused=solve_fused,**value))
    result=dict(scope='离线FP32运算顺序假设；不是实际Torch或硬件执行，不代替正式合同',case=case['case'],rows=rows)
    out=BATCH/'analysis/linalg_inv_ex/offline-rounding.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(rows,ensure_ascii=False))


if __name__=='__main__':main()
