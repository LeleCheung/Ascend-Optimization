"""读取msprof op simulator的公开CSV与逐核trace格式，不依赖KG/KGS。"""
import csv
import json
import math
from pathlib import Path
from statistics import median


def validate_time_unit(trace, instructions, unit):
    """用同核、同PC的CSV累计时间／cycles交叉核验trace单位，不读取显示单位。"""
    events=trace.get('traceEvents',[]) if isinstance(trace,dict) else trace
    names={e.get('pid'):e.get('args',{}).get('name') for e in events if e.get('ph')=='M' and e.get('name')=='process_name'}
    durations={}
    for event in events:
        args=event.get('args',{});duration=number(event.get('dur'))
        if event.get('ph')!='X' or 'pc_addr' not in args or not duration or names.get(event.get('pid'))=='CACHEMISS':continue
        key=pc(args['pc_addr']);durations[key]=durations.get(key,0)+duration
    ratios=[]
    for row in instructions:
        reference=row.get('cycles' if unit=='cycles' else 'running_time_us')
        if reference and row['pc'] in durations:ratios.append(durations[row['pc']]/reference)
    expected=1000 if unit=='ns' else 1
    matching=sum(abs(value/expected-1)<.05 for value in ratios)
    if len(ratios)<3 or matching/len(ratios)<.8:
        raise ValueError('trace单位无法由同PC的CSV核验：'+str(dict(unit=unit,compared=len(ratios),
            median_ratio=median(ratios) if ratios else None,expected_ratio=expected)))
    return dict(status='verified',unit=unit,compared_pcs=len(ratios),matching_pcs=matching,
                median_ratio=median(ratios),expected_ratio=expected,method='同PC的trace正时长总和对照CSV累计时间或cycles')


def pc(value):
    try:
        return hex(int(str(value),0))
    except (TypeError,ValueError):
        raise ValueError('无效指令地址：'+str(value)) from None


def number(value, integer=False):
    try:
        result=int(str(value),0) if integer else float(value)
    except (TypeError,ValueError):
        return None
    return result if math.isfinite(result) and result>=0 else None


def parse(directory, kernel, source, digest, case):
    """只读取指定kernel目录；原始逐核文件由采集器绑定，不混合多个核。"""
    roots=list(Path(directory).glob('OPPROF_*/simulator'))
    if len(roots)!=1:
        raise ValueError('模拟器报告目录缺失或不唯一')
    source=Path(source);lines=source.read_text(encoding='utf8').splitlines();cores=[]
    for path in sorted(roots[0].glob('*/*_instr_exe.csv')):
        raw=json.loads(path.with_name('trace.json').read_bytes())
        events=raw.get('traceEvents',[]) if isinstance(raw,dict) else raw
        by_pc={}
        for event in events:
            args=event.get('args',{})
            if event.get('ph')!='X' or 'pc_addr' not in args:
                continue
            key=pc(args['pc_addr'])
            if key not in by_pc or args.get('code'):
                by_pc[key]=event
        rows=[]
        with path.open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                key=pc(row['addr']);event=by_pc.get(key,{})
                item=dict(pc=key,instruction=row.get('instr') or event.get('name',''),
                    pipeline=row.get('pipe',''),calls=number(row.get('call_count'),True),
                    cycles=number(row.get('cycles'),True),running_time_us=number(row.get('running_time(us)')),
                    detail=row.get('detail',''))
                locations=[]
                for location in str(event.get('args',{}).get('code','')).splitlines():
                    file,sep,line=location.strip().rpartition(':')
                    if sep and line.isdecimal():
                        entry=dict(file=file,line=int(line))
                        if file==str(source) and 0<int(line)<=len(lines):entry['code']=lines[int(line)-1].strip()
                        locations.append(entry)
                if locations:
                    item['source']=next((v for v in reversed(locations) if v['file']==str(source)),locations[-1])
                    item['source_stack']=locations
                rows.append(item)
        if not rows:
            raise ValueError('模拟器指令CSV为空：'+str(path))
        cores.append(dict(name=path.parent.name,instructions=rows,listed_instruction_count=len(rows),
                          mapped_instruction_count=sum('source' in r for r in rows)))
    if not cores:
        raise ValueError('没有逐核指令CSV')
    return dict(profiler='msprof-op-simulator',source_sha256=digest,case=case,
                parser='hgkda public simulator CSV/trace adapter',kernels=[dict(name=kernel,cores=cores)])
