"""离线检验转置LU与两种三角求解顺序；不是设备评测。"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3];BATCH=PROJECT/'operators/campaigns/reference-closure-20261031'
def f32(x):
    try:return struct.unpack('f',struct.pack('f',x))[0]
    except OverflowError:return math.copysign(math.inf,x)
def transpose(a):return [list(x) for x in zip(*a)]
def inverse(matrix,layout,reciprocal,lu_fused,solve_fused,solve_mode):
    a=transpose(matrix) if layout=='transpose' else [r[:] for r in matrix];n=len(a);piv=[]
    lu_update=lambda x,y,z:f32(math.fma(-x,y,z)) if lu_fused else f32(z-f32(x*y))
    update=lambda x,y,z:f32(math.fma(-x,y,z)) if solve_fused else f32(z-f32(x*y))
    for j in range(n):
        p=max(range(j,n),key=lambda r:abs(a[r][j]));piv.append(p);a[j],a[p]=a[p],a[j];pivot=a[j][j]
        if pivot==0:return dict(info=j+1)
        rec=f32(1./pivot)
        for i in range(j+1,n):a[i][j]=f32(a[i][j]*rec) if reciprocal else f32(a[i][j]/pivot)
        for k in range(j+1,n):
            for i in range(j+1,n):a[i][k]=lu_update(a[i][j],a[j][k],a[i][k])
    b=[[float(i==j) for j in range(n)] for i in range(n)]
    if layout=='transpose' and solve_mode=='adjoint':
        # PA^T=LU -> A=U^T L^T P；依次解U^T、L^T、逆置换。
        for j in range(n):
            for k in range(n):
                value=b[j][k]
                for i in range(j):value=update(a[i][j],b[i][k],value)
                b[j][k]=f32(value/a[j][j])
        for j in range(n-1,-1,-1):
            for k in range(n):
                value=b[j][k]
                for i in range(j+1,n):value=update(a[i][j],b[i][k],value)
                b[j][k]=value
        for j in range(n-1,-1,-1):p=piv[j];b[j],b[p]=b[p],b[j]
    else:
        for j,p in enumerate(piv):b[j],b[p]=b[p],b[j]
        for j in range(n):
            for i in range(j+1,n):
                for k in range(n):b[i][k]=update(a[i][j],b[j][k],b[i][k])
        for j in range(n-1,-1,-1):
            for k in range(n):b[j][k]=f32(b[j][k]/a[j][j])
            for i in range(j):
                for k in range(n):b[i][k]=update(a[i][j],b[j][k],b[i][k])
        if layout=='transpose':b=transpose(b)
    return dict(info=0,inverse=b,lu_diagonal=[a[i][i] for i in range(n)])
def main():
    probe=BATCH/'analysis/linalg_inv_ex/contract-probe-fc75b4022866.json';raw=json.loads(probe.read_text(encoding='utf8'));rows=[]
    for case in raw['cases']:
        if not case.get('input',{}).get('values'):continue
        inputs=case['input']['values'];expected=case['outputs'][0]['npu_reference']['values']
        matrices=[inputs] if len(case['input']['shape'])==2 else inputs
        references=[expected] if len(case['input']['shape'])==2 else expected
        for index,(matrix,reference) in enumerate(zip(matrices,references)):
            for layout,reciprocal,fused,sfused,mode in itertools.product(['direct','transpose'],[False,True],[False,True],[False,True],['inverse_transpose','adjoint']):
                if layout=='direct' and mode=='adjoint':continue
                value=inverse(matrix,layout,reciprocal,fused,sfused,mode)
                if value['info']==0:
                    errors=[abs(x-y) for a,b in zip(value['inverse'],reference) for x,y in zip(a,b)]
                    matched=all(abs(x-y)<=case['tolerance']['atol']+case['tolerance']['rtol']*abs(y) for a,b in zip(value['inverse'],reference) for x,y in zip(a,b))
                    value.update(matched=matched,max_abs=max(errors),max_relative=max(abs(x-y)/max(abs(y),1.e-300) for a,b in zip(value['inverse'],reference) for x,y in zip(a,b)))
                value.pop('inverse',None)
                rows.append(dict(case=case['case'],matrix=index,layout=layout,reciprocal=reciprocal,lu_fused=fused,solve_fused=sfused,solve_mode=mode,**value))
    output=BATCH/'analysis/linalg_inv_ex/offline-layout-v2.json'
    result=dict(status='offline_arithmetic_hypotheses',probe=str(probe),probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),rows=rows,
        scope='Python显式FP32舍入模拟，不等于CPU BLAS或NPU实际路径；不能代替正确性、info和性能评测')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
    print(json.dumps(dict(rows=len(rows),matching_singular=[r for r in rows if 'correctness-1' in r['case'] and r.get('matched')]),ensure_ascii=False))
if __name__=='__main__':main()
