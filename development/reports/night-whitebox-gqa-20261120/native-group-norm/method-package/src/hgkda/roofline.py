"""同次 GEMM 采集的数值参考；经验包络与标称上界分开。"""
import math

def derive(facts,shape,dtype,empirical):
    if len(shape)!=3 or any(type(v) is not int or v<=0 for v in shape):raise ValueError('GEMM必须提供正整数M、N、K')
    if dtype not in ['float16','bfloat16','float32']:raise ValueError('缺少该精度的工作量合同')
    flops=2*math.prod(shape);duration=facts['duration_us']
    if type(duration) not in [int,float] or not math.isfinite(duration) or duration<=0:raise ValueError('无效同次采集时间')
    compute=[r['tflops'] for r in empirical['records'] if r['kind']=='gemm' and r['dtype']==dtype]
    copy=[empirical['gm_counter_reference']['gbps']] if empirical.get('gm_counter_reference') else []
    if not compute or not copy:raise ValueError('缺少同精度计算或同计数口径GM经验参考；不能混用逻辑复制字节')
    if any(type(v) not in [int,float] or not math.isfinite(v) or v<=0 for v in compute+copy):raise ValueError('经验参考包含无效值')
    compute=max(compute);bandwidth=max(copy)
    curves=[]
    for c in facts['curves']:
        if c['official_bytes'] is None or c['official_bytes']<=0:continue
        # Vector点保留本组件操作量，不能给每条流水重复分配整个GEMM的2MNK。
        curve=dict(c);curve['work_scope']='官方本组件内部操作量；不等于整个GEMM工作量'
        if 'Cube' in c['component'] or c['path'] in ['GM Read + Write','L2 Read + Write']:
            curve.update(algorithmic_flops=flops,algorithmic_tflops=flops/duration/1e6,
                hardware_over_algorithmic=c['official_flops']/flops,algorithmic_ai=flops/c['official_bytes'],
                work_scope='GEMM有效2MNK与官方内部操作数分别报告；不包含融合尾处理FLOPs')
        if c['path'] in ['GM Read + Write','L2 Read + Write']:
            bw=bandwidth if c['path']=='GM Read + Write' else c['bandwidth_tbps']*1000
            compute_roof=compute
            bound=min(compute_roof,curve['algorithmic_ai']*bw/1000)
            curve.update(reference_compute_tflops=compute_roof,reference_bandwidth_gbps=bw,
                reference_bound_tflops=bound,reference_achievement=curve['algorithmic_tflops']/bound,
                limiting_side='memory' if curve['algorithmic_ai']*bw/1000<compute_roof else 'compute',
                reference_kind='经验观察包络；不是已证明物理上界' if c['path']=='GM Read + Write' else '官方L2带宽模型＋同精度经验有效计算参考')
        curves.append(curve)
    issues=[]
    if not curves:issues.append('没有有效官方Roofline数据；拒绝用零计数构图')
    if any(c.get('reference_achievement',0)>1.05 for c in curves):issues.append('超过参考线；检查计数、时间和上限，不裁剪数值')
    if any(c['path']=='GM Read + Write' and c['bandwidth_tbps']*1000<bandwidth*.95 for c in curves):
        issues.append('官方GM常量低于同PMU模型persistent copy观测，保留原值但不宣称其为本机已校准物理上界')
    return dict(algorithmic_flops=flops,algorithmic_tflops=flops/duration/1e6,duration_us=duration,
        curves=curves,issues=issues,model_scope='同次profiling的官方PMU派生点；GM经验包络与标称模型分别报告',
        diagnosis='先比较GM与L2差距；远低于包络不直接证明带宽受限，结合受控分块、等待和IR查流水/资源压力')
