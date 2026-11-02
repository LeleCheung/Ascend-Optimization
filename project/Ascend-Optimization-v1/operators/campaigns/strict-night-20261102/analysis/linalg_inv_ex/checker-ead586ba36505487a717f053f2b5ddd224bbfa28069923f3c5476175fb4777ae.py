"""离线检验OpenBLAS按列更新LU的运算顺序；不运行设备、不生成候选常量。"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct


def f32(x):
    try:return struct.unpack('f',struct.pack('f',x))[0]
    except OverflowError:return math.copysign(math.inf,x)


def dot(a,b,mode):
    value=0.
    for x,y in zip(a,b):
        value=f32(math.fma(x,y,value)) if mode=='fma' else f32(value+f32(x*y))
    return value


def subtract_products(value,a,b,mode):
    if mode.startswith('dot_'):return f32(value-dot(a,b,mode[4:]))
    for x,y in zip(a,b):
        value=f32(math.fma(-x,y,value)) if mode=='update_fma' else f32(value-f32(x*y))
    return value


def inverse(matrix,layout,reciprocal,upper_dot,gemv,solve):
    a=[list(r) for r in zip(*matrix)] if layout=='transpose' else [r[:] for r in matrix]
    n=len(a);piv=[]
    # OpenBLAS getf2_k.c：当前列应用先前置换，先计算上三角，再一次更新剩余列段。
    for j in range(n):
        for k,p in enumerate(piv):a[k][j],a[p][j]=a[p][j],a[k][j]
        for i in range(1,j):a[i][j]=f32(a[i][j]-dot(a[i][:i],[a[k][j] for k in range(i)],upper_dot))
        for i in range(j,n):
            a[i][j]=subtract_products(a[i][j],a[i][:j],[a[k][j] for k in range(j)],gemv)
        p=max(range(j,n),key=lambda i:abs(a[i][j]));piv.append(p)
        for k in range(j+1):a[j][k],a[p][k]=a[p][k],a[j][k]
        pivot=a[j][j]
        if pivot==0:return dict(info=j+1,lu=a)
        rec=f32(1./pivot)
        for i in range(j+1,n):a[i][j]=f32(a[i][j]*rec) if reciprocal else f32(a[i][j]/pivot)
    b=[[float(i==k) for k in range(n)] for i in range(n)]
    if layout=='transpose':
        # PA^T=LU；解U^T、L^T，再执行逆置换，保持实际求逆入口的adjoint方向。
        for j in range(n):
            for k in range(n):
                value=subtract_products(b[j][k],[a[i][j] for i in range(j)],[b[i][k] for i in range(j)],solve)
                b[j][k]=f32(value/a[j][j])
        for j in range(n-1,-1,-1):
            for k in range(n):
                b[j][k]=subtract_products(b[j][k],[a[i][j] for i in range(j+1,n)],[b[i][k] for i in range(j+1,n)],solve)
        for j in range(n-1,-1,-1):p=piv[j];b[j],b[p]=b[p],b[j]
    else:
        for j,p in enumerate(piv):b[j],b[p]=b[p],b[j]
        for j in range(n):
            for k in range(n):
                b[j][k]=subtract_products(b[j][k],a[j][:j],[b[i][k] for i in range(j)],solve)
        for j in range(n-1,-1,-1):
            for k in range(n):
                value=subtract_products(b[j][k],a[j][j+1:],[b[i][k] for i in range(j+1,n)],solve)
                b[j][k]=f32(value/a[j][j])
    return dict(info=0,inverse=b,lu=a)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists(),'保留已冻结离线结果'
    data=json.loads(args.probe.read_text(encoding='utf8'));rows=[]
    modes=['dot_split','dot_fma','update_split','update_fma']
    for case in data['cases']:
        if not case.get('input',{}).get('values'):continue
        inputs=case['input']['values'];expected=case['outputs'][0]['npu_reference']['values']
        matrices=[inputs] if len(case['input']['shape'])==2 else inputs
        targets=[expected] if len(case['input']['shape'])==2 else expected
        for index,(matrix,target) in enumerate(zip(matrices,targets)):
            for layout,reciprocal,upper,gemv,solve in itertools.product(['direct','transpose'],[False,True],['split','fma'],modes,modes):
                result=inverse(matrix,layout,reciprocal,upper,gemv,solve)
                row=dict(case=case['case'],matrix=index,layout=layout,reciprocal=reciprocal,
                         upper_dot=upper,gemv=gemv,solve=solve,info=result['info'])
                row['lu_diagonal']=[result['lu'][i][i] for i in range(len(matrix))]
                if result['info']==0:
                    pairs=[(x,y) for a,b in zip(result['inverse'],target) for x,y in zip(a,b)]
                    row['matched']=all(abs(x-y)<=case['tolerance']['atol']+case['tolerance']['rtol']*abs(y) for x,y in pairs)
                    row['max_abs']=max(abs(x-y) for x,y in pairs)
                    row['max_relative']=max(abs(x-y)/max(abs(y),1.e-300) for x,y in pairs)
                rows.append(row)
    value=dict(status='offline_leftlooking_hypotheses',probe=str(args.probe),
               probe_sha256=hashlib.sha256(args.probe.read_bytes()).hexdigest(),
               checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows,
               scope='显式FP32舍入的按列更新模型；未重现CPU微内核或NPU设备执行，不是正式成绩，不把固定输入写进候选。')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    summary={}
    for r in rows:
        key=r['case']+':'+str(r['matrix']);item=summary.setdefault(key,dict(attempted=0,matched=0,matching_modes=[]))
        item['attempted']+=1
        if r.get('matched'):
            item['matched']+=1
            if len(item['matching_modes'])<3:item['matching_modes'].append({k:r[k] for k in ['layout','reciprocal','upper_dot','gemv','solve']})
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
