"""读取固定版本可视化JSON分段，区分CSV字段与官方Roofline派生口径。"""
import csv,hashlib,json,math
from pathlib import Path

def number(value):
    try:
        result=float(value)
        return result if math.isfinite(result) else None
    except (TypeError,ValueError):return None

def visual_sections(path):
    data=path.read_bytes();position=0;sections=[]
    # 本机格式：uint64小端payload长度、uint32类型、UTF8 JSON和零填充。
    while position<len(data):
        if len(data)-position<12:raise ValueError('可视化段头不完整')
        size=int.from_bytes(data[position:position+8],'little');tag=data[position+8:position+12].hex()
        position+=12
        if not 0<size<=len(data)-position:raise ValueError('非法可视化段长度')
        raw=data[position:position+size].rstrip(b'\0');position+=size
        value=json.loads(raw.decode('utf8'))
        sections.append(dict(tag=tag,data=value))
    return sections

def table(directory,name):
    files=list(Path(directory).rglob(name))
    if len(files)!=1:raise ValueError('附件不唯一：'+name)
    with files[0].open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))

def parse(directory):
    directory=Path(directory);basics=table(directory,'OpBasicInfo.csv')
    if len(basics)!=1:raise ValueError('基本信息不唯一')
    basic=basics[0];duration=number(basic['Task Duration(us)'])
    if duration is None or duration<=0:raise ValueError('无效设备时间')
    visual=list(directory.rglob('visualize_data.bin'))
    if len(visual)!=1:raise ValueError('可视化附件不唯一')
    sections=visual_sections(visual[0]);roof=[s['data'] for s in sections if 'multiple_rooflines' in s['data']]
    identity=[s['data'] for s in sections if 'block_detail' in s['data'] or 'mix_block_detail' in s['data']]
    if len(roof)!=1 or len(identity)!=1:raise ValueError('Roofline或身份段不唯一')
    identity=identity[0]
    if identity['name']!=basic['Op Name'] or str(identity['device_id'])!=basic['Device Id'] or not math.isclose(identity['duration'],duration,rel_tol=2e-6):
        raise ValueError('CSV与可视化采集身份不符')
    if basic['Device Id']!='7':raise ValueError('非卡7采集')
    curves=[];unsupported=[]
    for group in roof[0]['multiple_rooflines']:
        for line in group['rooflines']:
            x,y=map(number,line['point'])
            bw=number(line.get('bw'));compute=number(line.get('computility'))
            if any(v is None or v<=0 for v in [x,y,bw,compute]):
                unsupported.append(dict(component=group['title'],path=line.get('bw_name'),
                    reason='缺失、非有限或非正数；纯复制不生成FLOPs屋顶点',raw=line))
                continue
            flops=y*duration*1e6
            curves.append(dict(component=group['title'],path=line['bw_name'],
                compute_path=line['computility_name'],compute_roof_tflops=compute,
                bandwidth_tbps=bw,arithmetic_intensity=x,performance_tflops=y,
                official_flops=flops,official_bytes=flops/x if x>0 else None,reported_ratio=line.get('ratio'),
                source='固定本机Msopprof可视化官方派生值；非Memory.csv主存字段'))
    arithmetic=table(directory,'ArithmeticUtilization.csv')
    def total(field):
        values=[number(row.get(field)) for row in arithmetic];values=[n for n in values if n is not None]
        return sum(values) if values else None
    cache=table(directory,'L2Cache.csv');cache_counts={}
    for prefix in ['aic','aiv']:
        for event in ['write_cache_hit','write_cache_miss_allocate','r0_read_cache_hit','r0_read_cache_miss_allocate','r1_read_cache_hit','r1_read_cache_miss_allocate']:
            field=prefix+'_'+event;values=[number(r.get(field)) for r in cache];values=[v for v in values if v is not None]
            if values:cache_counts[field]=sum(values)
    pipe_rows=table(directory,'PipeUtilization.csv');pipeline={}
    for field in pipe_rows[0] if pipe_rows else []:
        if not field.endswith(('_ratio','_time(us)')):continue
        values=[number(r.get(field)) for r in pipe_rows];values=[v for v in values if v is not None]
        if values:pipeline[field]=dict(mean=sum(values)/len(values),minimum=min(values),maximum=max(values),count=len(values))
    memories=table(directory,'Memory.csv');memory={}
    for field in ['read_main_memory_datas(KB)','write_main_memory_datas(KB)','GM_to_L1_datas(KB)','GM_to_UB_datas(KB)','UB_to_GM_datas(KB)']:
        values=[number(r.get(field)) for r in memories];values=[v for v in values if v is not None]
        memory[field]=sum(values)*1024 if values else None
    return dict(basic=basic,duration_us=duration,curves=curves,unsupported_curves=unsupported,csv_cube_fops=total('aic_cube_fops'),
                csv_cube_instructions=total('aic_cube_fp_instr_number'),memory_csv_bytes=memory,cache_counts=cache_counts,pipeline=pipeline,
                artifacts=[dict(path=str(f.relative_to(directory)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size)
                    for f in sorted(directory.rglob('*')) if f.is_file() and f.suffix in ['.csv','.bin']])
