"""把已核验的数值证据压缩为可注入KG角色的冻结摘要。"""
import argparse,hashlib,json,math
from pathlib import Path

def compact(path):
    raw=Path(path).read_bytes();data=json.loads(raw)
    if data.get('status')!='completed' or not data.get('cases'):raise ValueError('分析尚未完成')
    cases=[]
    for row in data['cases']:
        spec=row['spec'];facts=row['facts'];diagnostic=row['diagnostic']
        if len(spec['source_sha256'])!=64 or facts['basic']['Device Id']!='7':raise ValueError('缺少源码或设备身份')
        paths=[]
        for c in diagnostic['curves']:
            if c['path'] not in ['GM Read + Write','L2 Read + Write']:continue
            fields=['component','path','official_bytes','official_flops','hardware_over_algorithmic',
                'algorithmic_ai','algorithmic_tflops','reference_compute_tflops','reference_bandwidth_gbps',
                'reference_bound_tflops','reference_achievement','limiting_side','reference_kind']
            value={k:c[k] for k in fields}
            if any(isinstance(v,float) and not math.isfinite(v) for v in value.values()):raise ValueError('非有限诊断')
            paths.append(value)
        if not paths:raise ValueError('缺少有效GM/L2点')
        pipeline={k:v for k,v in facts.get('pipeline',{}).items() if k in ['aic_cube_ratio','aic_mte1_ratio',
            'aic_mte2_ratio','aiv_vec_ratio','aiv_scalar_ratio','aic_scalar_ratio','aic_scalar_mte1_stall_time(us)']}
        cases.append(dict(id=row['id'],spec=spec,kernel=facts['basic']['Op Name'],duration_us=facts['duration_us'],
            algorithmic_flops=diagnostic['algorithmic_flops'],paths=paths,pipeline=pipeline,issues=diagnostic['issues']))
    return dict(schema_version='hgkda-roofline-v1',source=dict(path=str(Path(path).resolve()),
        sha256=hashlib.sha256(raw).hexdigest()),cases=cases,
        rules=['PyTorch侧仅GEMM组件，候选为融合kernel；完整速度由KGS裁决。',
               'GM字节来自官方cache-miss/eviction模型，不能换成Memory.csv接口流量或逻辑字节。',
               'GM参考和矩阵点使用相同cache事件模型；逻辑copy吞吐单列。同精度经验线非物理峰值，官方0.8TB/s常量匹配未决。',
               'Vector内部操作不分配整个2MNK；FP32内部工作量与有效算法量分别报告。',
               '模型的limiting_side只表示参考线哪侧更低，不证明当前kernel计算受限。低达成率仍需IR/流水与受控实验区分原因。'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('保留已有证据，请指定新路径')
    result=compact(a.analysis);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(cases=len(result['cases']),bytes=a.output.stat().st_size)))

if __name__=='__main__':main()
