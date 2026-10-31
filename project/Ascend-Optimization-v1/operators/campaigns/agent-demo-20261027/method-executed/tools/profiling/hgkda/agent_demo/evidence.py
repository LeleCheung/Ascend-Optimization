"""组织真实成对差异、IR、Roofline与热点，给模型提供可查证输入。"""
import argparse,hashlib,importlib.util,json,re,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3]

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);a=p.parse_args();b=a.batch.resolve()
    assert not (b/'evidence-qualified.json').exists()
    sys.path.insert(0,str(HERE.parent/'roofline'))
    from profile_data import parse
    from analyze import load_reference,derive
    reference=PROJECT/'operators/campaigns/roofline-20261027';empirical=load_reference(reference)
    pairs=[];hotspots=None
    for size in [512,1024]:
        original=b/'seed-profile';records={}
        comparison=b/'paired'/('float16-'+str(size))/'comparison.json';full=json.loads(comparison.read_bytes())
        for side in ['candidate','pytorch']:
            directory=original/(side+'-float16-'+str(size));spec=json.loads((directory/'spec.json').read_bytes());facts=parse(directory/'profile')
            driver=json.loads((directory/'compiled/driver.json').read_bytes());discovery=json.loads((directory/'discovery.json').read_bytes())
            assert driver['status']=='passed' and driver['spec']==spec and facts['basic']['Device Id']=='7'
            assert discovery['selected']==[facts['basic']['Op Name']]
            records[side]=dict(spec=spec,basic=facts['basic'],duration_us=facts['duration_us'],
                roofline=derive(facts,spec['shape'],spec['dtype'],empirical),pipeline=facts['pipeline'],
                memory_csv_bytes=facts['memory_csv_bytes'],compiled=driver['compiled'])
        picked=[row for row in full['differences'] if any(k in row['metric'].lower() for k in ['mte','scalar','vec','cube','wait','cflt','conflict'])]
        pairs.append(dict(size=size,records=records,differences=picked,derived=full['derived_differences'],full_comparison=str(comparison),
            comparison_sha256=hashlib.sha256(comparison.read_bytes()).hexdigest()))
    ins=b/'seed-instruction-v3/summary.json'
    if ins.exists():
        hotspots=json.loads(ins.read_bytes())
        log=(ins.parent/'stdout.txt').read_text(errors='replace')
        child_codes=[int(x) for x in re.findall(r'Child process exited with status (\d+)',log)]
        if any(code!=0 for code in child_codes):
            hotspots=dict(hotspots,status='partial',evidence_valid=hotspots['instructions']>0,
                child_exit_codes=child_codes,qualification='驱动完成数值检查，导出指令可读，但子进程退出异常；热点为部分证据，不记完整模拟器成功')
    ir=[]
    for path in (b/'seed-profile/candidate-float16-1024/compiled').rglob('*.txt'):
        if path.name not in ['ttir.txt','bcmlir.txt']:continue
        lines=path.read_text(encoding='utf8').splitlines()
        excerpts=[dict(line=i+1,text=text.strip()) for i,text in enumerate(lines) if any(x in text for x in ['linalg.matmul','tt.dot','memref.copy','memref.alloc','scf.for'])]
        ir.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),static_occurrences=excerpts))
    knowledge=HERE.parent/'skills/hgkda/references/性能指标/矩阵乘复用方向与流水验证.md'
    result=dict(status='completed',operator='matmul_bias_activation',source_sha256=hashlib.sha256((b/'seed.py').read_bytes()).hexdigest(),
        pairs=pairs,ir=ir,instruction=hotspots,reference_batch=str(reference),reference_kind='同环境同精度经验参考，非物理峰值',
        input_contract=dict(seed=20261027,shape_order='M,N,K',dtype='float16',driver_sha256=hashlib.sha256((HERE.parent/'roofline/matrix_driver.py').read_bytes()).hexdigest()),
        knowledge=dict(path=str(knowledge),sha256=hashlib.sha256(knowledge.read_bytes()).hexdigest(),prior='已知历史优化知识，允许使用，不称为盲发现'),
        limits=['PyTorch指标只采GEMM组件，候选为融合kernel；完整算子由KGS计时',
            '静态IR出现次数不是动态指令数','模拟器时间不是真机耗时','流水比例重叠，不相加为总耗时','官方GM事件模型不是Memory.csv接口字节'])
    if (b/'evidence.json').exists():
        save(b/'evidence-qualified.json',result)
    else:save(b/'evidence.json',result)
    compact=dict(result);compact['pairs']=[]
    for pair in pairs:
        q=dict(pair);q['records']={s:{k:v for k,v in row.items() if k!='compiled'} for s,row in pair['records'].items()};compact['pairs'].append(q)
    save(b/'agent-input.json',compact)
    print(json.dumps(dict(status='completed',pairs=len(pairs),instruction_status=hotspots.get('status') if hotspots else 'missing')),flush=True)

if __name__=='__main__':main()
