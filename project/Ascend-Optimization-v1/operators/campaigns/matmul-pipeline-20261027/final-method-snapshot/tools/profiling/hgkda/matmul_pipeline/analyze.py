"""把本批真实延迟、同输入流量和IR差异关联为可核验机制报告。"""
import argparse,hashlib,json,re,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3]
sys.path.insert(0,str(HERE.parent/'roofline'))
from profile_data import parse

def ir_summary(directory):
    outputs=[]
    for f in directory.rglob('*.txt'):
        if f.name not in ['bcmlir.txt','ttir.txt']:continue
        lines=f.read_text(encoding='utf8').splitlines();matches=[dict(line=i+1,text=x.strip()) for i,x in enumerate(lines)
            if re.search(r'linalg.matmul|tt.dot|memref.alloc|memref.copy|scf.for|arith.*cast|tensor<.*xf32>',x)]
        outputs.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),kind=f.name,
            static_matmul_occurrences=sum('linalg.matmul' in x or 'tt.dot' in x for x in lines),
            copies=sum('memref.copy' in x for x in lines),allocations=sum('memref.alloc' in x for x in lines),excerpts=matches,
            scope='每种IR分别计静态文本出现；不等于动态指令数、实际UB大小或关键路径'))
    return outputs

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('使用新分析目录')
    a.output.mkdir(parents=True);b=a.batch;comparison=json.loads((b/'validation/comparison.json').read_bytes())
    pairs=[]
    for size in [1024,2048]:
        records={}
        for side in ['parent','candidate']:
            root=b/(side+'-profile')/(side+'-float16-'+str(size));spec=json.loads((root/'spec.json').read_bytes());facts=parse(root/'profile')
            driver=json.loads((root/'compiled/driver.json').read_bytes());discovery=json.loads((root/'discovery.json').read_bytes())
            if driver.get('status')!='passed' or driver['spec']!=spec or discovery['selected']!=[facts['basic']['Op Name']]:raise ValueError('采集身份不符')
            records[side]=dict(spec=spec,facts=facts,ir=ir_summary(root/'compiled'))
        def path(side,name):return next(c for c in records[side]['facts']['curves'] if c['path']==name)['official_bytes']
        ratios={name:path('candidate',name)/path('parent',name) for name in ['GM Read + Write','L2 Read + Write','Read from L1','Write to L1']}
        pairs.append(dict(size=size,dtype='float16',records=records,traffic_candidate_over_parent=ratios,
            profiling_speedup=records['parent']['facts']['duration_us']/records['candidate']['facts']['duration_us'],
            interpretation='检查复用方向是否改变各层流量及编译形状；时间与计数同次，活跃均值不跨不同program数当作整卡利用率'))
    analysis=dict(status='completed',comparison=comparison,pairs=pairs,
        attribution='Codex实现通用变换并编排；CC＋DeepSeek读取Roofline与开发轨迹选择追加配置。非纯原生KG独立优化。',
        rejected_model_claims=['回执称BF16大K回父版与实际模板不符：新路径适用于满足整除条件的BF16大K。',
            '回执称默认调度接近上限缺少物理峰值与关键路径证据，不纳入结论。',
            'group4为分组遍历，不是蛇形；不以模型表述替代源码。'])
    (a.output/'mechanism.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='completed',pairs=len(pairs),selected=comparison['selected'],speedup=comparison['actual_candidate_over_parent'])))

if __name__=='__main__':main()
