#!/usr/bin/env python3
"""复核已采集的官方 Roofline 文本与计数 CSV，不重新占用设备。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics


def summarize(artifacts: Path, source: Path) -> dict:
    request = json.loads((artifacts/'request.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if request['source_sha256'] != digest:
        raise ValueError('profile 请求与候选源码 SHA 不匹配')
    case = request['case_id']
    if case != 'benchmark/test_matmul_bias_activation.py::test_matmul_bias_activation::core::float16::2':
        raise ValueError('此脚本只解释已核验的 1024³ fp16 case')
    roots = list(artifacts.rglob('OpBasicInfo.csv'))
    if len(roots) != 1:
        raise ValueError('采集必须只有一份 kernel 基本信息')
    root = roots[0].parent
    def read(name):
        with (root/name).open(encoding='utf-8-sig',newline='') as f:
            return list(csv.DictReader(f))
    info = read('OpBasicInfo.csv')
    if len(info) != 1 or info[0]['Op Name'] != 'matmul_bias_activation_kernel_mix_aic':
        raise ValueError('kernel 身份不匹配')
    if info[0]['Device Id'] != '7':
        raise ValueError('采集不在本轮指定的物理卡 7')
    text = (artifacts/'stdout.txt').read_text(encoding='utf-8',errors='replace')
    match = re.search(r'latency bound:\s*([^\r\n]+)',text)
    pipe = read('PipeUtilization.csv')
    cube = [r for r in pipe if r['sub_block_id']=='cube0']
    if len(cube) != int(info[0]['Block Dim']):
        raise ValueError('CSV 缺少部分 program 的记录')
    ratios = {}
    for name in ['aic_cube_ratio','aic_mte1_ratio','aic_mte2_ratio','aic_scalar_ratio']:
        values = [float(r[name]) for r in cube]
        ratios[name] = {'mean':statistics.mean(values),'min':min(values),'max':max(values)}
    counter_fops = sum(float(r['aic_cube_fops']) for r in read('ArithmeticUtilization.csv')
                       if r['sub_block_id']=='cube0')
    expected_fops = 2*1024**3
    manifest = {str(f.relative_to(artifacts)):{'bytes':f.stat().st_size,
                'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
                for f in [artifacts/'request.json',artifacts/'stdout.txt',*root.glob('*.csv')]}
    return {'case_id':case,'source_sha256':digest,'flaggems_revision':request['flaggems_revision'],
            'kernel':info[0], 'official_latency_bound':match.group(1).strip() if match else None,
            'cube_programs':len(cube),'pipeline_ratios':ratios,
            'raw_counter_fops_sum':counter_fops,'algorithm_gemm_flops':expected_fops,
            'counter_to_algorithm_flops_ratio':counter_fops/expected_fops,
            'roofline':{'status':'official_qualitative_diagnosis_available',
                        'numeric_roof_status':'unavailable',
                        'reason':'原始 FOP 计数总和与算法 FLOPs 不同；计数缩放、有效 Bytes 和 roof 尚未核验'},
            'next_optimization':'减少 K 循环和搬运同步；按片上容量选择 tile，启用可通过正确性门禁的流水线。',
            'artifacts':manifest}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    report=summarize(a.artifacts,a.source)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    ratios=report['pipeline_ratios']
    lines=['# matmul_bias_activation 官方 Roofline 采集解读','',
           'Msopprof 给出 **latency bound: memory caused**。这是该工具对本 case 的定性诊断；完整数值 Roofline 尚未建立。','',
           f"固定 master、1024³ fp16、物理卡 7；kernel `{report['kernel']['Op Name']}`；采集 task duration 为 {report['kernel']['Task Duration(us)']} μs。",'',
           '| 指标 | program 记录均值 |','| --- | ---: |']
    lines += [f'| {name} | {values["mean"]:.2%} |' for name,values in ratios.items()]
    lines += ['', '上述均值用于描述 CSV，流水线比例不能相加，也不代表 HBM 峰值利用率。', '',
              f"原始 Cube FOP 计数总和 {report['raw_counter_fops_sum']:.0f}，算法 GEMM FLOPs {report['algorithm_gemm_flops']}，相差 {1/report['counter_to_algorithm_flops_ratio']:.0f} 倍。需要先确认计数单位和缩放，才能画可信的数值 Roofline。",'',
              '优化方向：增加 K 分块、减少循环与搬运同步，启用 multibuffer/unit_flag；宽 tile 若超出 UB 容量则缩小分块。效果以完整评测结果为准。','',
              '早期 summary.json 仅按文件名寻找 roofline，漏掉了 stdout 中的官方结论。该原始 summary 保留，本报告补充正确解释。','']
    a.output.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'official_latency_bound':report['official_latency_bound'],
                      'numeric_roof_status':report['roofline']['numeric_roof_status']},ensure_ascii=False))


if __name__=='__main__':
    main()
