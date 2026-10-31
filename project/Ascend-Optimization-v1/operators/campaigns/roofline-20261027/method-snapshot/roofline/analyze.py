"""校准后生成HG-KDA数值Roofline证据；经验参考与官方模型分开。"""
import argparse,hashlib,importlib.util,json,math,sys,os
from pathlib import Path
from profile_data import parse

HERE=Path(__file__).resolve().parent

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def load_reference(batch):
    empirical=json.loads((batch/'empirical-v2/results.json').read_bytes())
    directory=batch/'calibration-v2/copy-large-persistent'
    spec=json.loads((directory/'spec.json').read_bytes());facts=parse(directory/'profile')
    gm=next(c for c in facts['curves'] if c['path']=='GM Read + Write')
    empirical['gm_counter_reference']=dict(gbps=gm['official_bytes']/facts['duration_us']/1000,
        bytes=gm['official_bytes'],duration_us=facts['duration_us'],source=str(directory),
        source_sha256=hashlib.sha256((directory/'spec.json').read_bytes()).hexdigest(),
        artifacts=facts['artifacts'],scope='与矩阵点相同官方GM cache模型；单次persistent copy观测，不是已证明峰值',
        logical_bytes=spec['elements']*4,logical_gbps=spec['elements']*4/facts['duration_us']/1000)
    return empirical

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

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile-directories',nargs='+',default=['matrix-profile']);p.add_argument('--plot',action='store_true');a=p.parse_args()
    if a.output.exists():raise ValueError('保留已有分析，使用新目录')
    if os.name=='nt':a.output=Path('\\\\?\\'+str(a.output.resolve()))
    empirical=load_reference(a.batch);a.output.mkdir(parents=True)
    tool=importlib.util.spec_from_file_location('teacher_roofline',HERE/'teacher_roofline.py');teacher=importlib.util.module_from_spec(tool);sys.modules[tool.name]=teacher;tool.loader.exec_module(teacher)
    records=[];datasets={}
    for label in a.profile_directories:
      for directory in sorted((a.batch/label).iterdir()):
        if not directory.is_dir() or not (directory/'spec.json').is_file():continue
        spec=json.loads((directory/'spec.json').read_text(encoding='utf8'));facts=parse(directory/'profile')
        captured=json.loads((directory/'compiled/driver.json').read_bytes())
        discovery=json.loads((directory/'discovery.json').read_bytes())
        if captured.get('status')!='passed' or captured.get('spec')!=spec:raise ValueError('输入spec与采集驱动身份不符')
        if discovery.get('selected')!=[facts['basic']['Op Name']]:raise ValueError('选定kernel与采集身份不符')
        diagnostic=derive(facts,spec['shape'],spec['dtype'],empirical)
        identity=label+'/'+directory.name;records.append(dict(id=identity,spec=spec,facts=facts,diagnostic=diagnostic))
        save(a.output/'facts'/(identity.replace('/','-')+'.json'),records[-1])
        gm=next((c for c in diagnostic['curves'] if c['path']=='GM Read + Write'),None)
        if gm is None:continue
        datasets.setdefault(spec['dtype'],[]).append(dict(case_id=identity.replace('/','-'),op='matmul_bias_activation-'+spec['side'],dtype=spec['dtype'],status='passed',
            metrics=dict(flops=diagnostic['algorithmic_flops'],bytes=gm['official_bytes'],duration_us=facts['duration_us'],scope='single-kernel-profile',kernel_count=1),
            sources=dict(flops='algorithmic_2MNK_excludes_epilogue',bytes='hardware_counter_official_GM_cache_model',time='same_capture_device_counter',source_path=str(directory)),
            capture=dict(interval_match=True,duration_ratio=1.0),kernels=[dict(name=facts['basic']['Op Name'],duration_us=facts['duration_us'])],
            metadata=dict(side=spec['side'],note='PyTorch为GEMM组件；候选为融合kernel。不是完整算子速度对比。',
                reference_kind=gm['reference_kind'],physical_byte_model='官方cache miss与eviction派生，非Memory.csv接口字节')))
    for dtype,points in datasets.items():
        directory=a.output/'empirical-reference'/dtype;directory.mkdir(parents=True)
        compute=max(r['tflops'] for r in empirical['records'] if r['kind']=='gemm' and r['dtype']==dtype)
        bandwidth=empirical['gm_counter_reference']['gbps']
        manifest=dict(schema_version='1.0',dataset_id='roofline-20261027-'+dtype,
            platform=dict(vendor='huawei',device='Ascend910B4-1',implementation='HG-KDA'),
            benchmark=dict(name='同输入真机采集，PyTorch组件单列'),profiler=dict(name='固定本机msprof op',version='CANN9.0.0'),
            compute_roofs={dtype:dict(compute_path='same_dtype_empirical_algorithmic',tflops=compute,source='empirical_observed_reference_not_physical_upper_bound')},
            bandwidth_roof=dict(memory_level='GM官方cache模型',gbps=bandwidth,source='same_PMU_model_persistent_copy_reference_not_hardware_peak'),
            scope='老师分析器的roof字段在本数据集中解释为经验参考线；达成率不是物理峰值利用率')
        save(directory/'manifest.json',manifest)
        (directory/'points.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in points),encoding='utf8')
        results=teacher.analyze(teacher.load_points(directory/'points.jsonl',teacher.load_manifest(directory/'manifest.json')),1.05)
        teacher.write_outputs(results,directory/'teacher-output',a.plot,manifest)
    save(a.output/'diagnosis.json',dict(status='completed',cases=records,empirical_reference=empirical,
        teacher_source=json.loads((HERE/'teacher-source.json').read_text(encoding='utf8')),
        important='数值Roofline点、官方component模型和经验参考已接入；HBM标称上界未经独立核实，经验包络不是已证明物理上限'))
    print(json.dumps(dict(cases=len(records),datasets=list(datasets)),ensure_ascii=False))

if __name__=='__main__':main()
