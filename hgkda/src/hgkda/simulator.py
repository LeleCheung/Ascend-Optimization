"""Read public msprof op simulator CSV and per-core trace formats without KG/KGS dependencies."""
import csv
import json
import math
import re
from pathlib import Path
from statistics import median


def validate_time_unit(trace, instructions, unit):
    """Cross-check trace units against cumulative CSV time/cycles for the same core and PC, ignoring display units."""
    events=trace.get('traceEvents',[]) if isinstance(trace,dict) else trace
    names={e.get('pid'):e.get('args',{}).get('name') for e in events if e.get('ph')=='M' and e.get('name')=='process_name'}
    durations={};intervals={}
    for event in events:
        args=event.get('args',{});duration=number(event.get('dur'))
        if event.get('ph')!='X' or 'pc_addr' not in args or not duration or names.get(event.get('pid'))=='CACHEMISS':continue
        key=pc(args['pc_addr']);durations[key]=durations.get(key,0)+duration
        start=number(event.get('ts'))
        if start is not None:intervals.setdefault(key,[]).append((start,start+duration))
    # CSV runtime merges overlapping intervals for a PC; cycles retain cumulative execution cycles.
    # Legacy offline samples without timestamps still use summed durations.
    if unit!='cycles':
        for key,spans in intervals.items():
            merged=[]
            for start,end in sorted(spans):
                if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
                else:merged.append([start,end])
            durations[key]=sum(end-start for start,end in merged)
    ratios=[];excluded_sync=[];scalar_ratios=[]
    for row in instructions:
        reference=row.get('cycles' if unit=='cycles' else 'running_time_us')
        # CSV synchronization waits may include queue time; traces show actual wait intervals.
        # Retain them for hotspot-region and timeline analysis, but exclude them from time-unit calibration.
        if re.fullmatch(r'BAR|SET_FLAG|WAIT_FLAG',row.get('instruction',''),re.I):
            excluded_sync.append(row['pc']);continue
        if unit!='cycles' and (not reference or reference<.1):continue
        if reference and row['pc'] in durations:
            ratio=durations[row['pc']]/reference
            # SCALAR single-cycle instructions may use cycle conversion in CSV and nanosecond quantization in traces.
            # Cross-check units with independent Vector/MTE instructions; retain Scalar differences instead of claiming exact agreement.
            if unit!='cycles' and row.get('pipeline','').upper() in {'SCALAR','SCALARLDST','FLOWCTRL'}:
                scalar_ratios.append(ratio)
            else:ratios.append(ratio)
    expected=1000 if unit=='ns' else 1
    matching=sum(abs(value/expected-1)<.05 for value in ratios)
    if len(ratios)<3 or matching/len(ratios)<.8:
        raise ValueError('Trace unit cannot be verified against same-PC CSV records: '+str(dict(unit=unit,compared=len(ratios),
            median_ratio=median(ratios) if ratios else None,expected_ratio=expected)))
    return dict(status='verified',unit=unit,compared_pcs=len(ratios),matching_pcs=matching,
                median_ratio=median(ratios),expected_ratio=expected,
                excluded_sync_pcs=excluded_sync,
                scalar_comparison=dict(compared_pcs=len(scalar_ratios),median_ratio=median(scalar_ratios) if scalar_ratios else None,
                    matching_pcs=sum(abs(value/expected-1)<.05 for value in scalar_ratios),
                    scope='Retain Scalar timing-scope/quantization differences without using them for unit calibration; timelines use tool-reported intervals'),
                method='Compare same-PC non-Scalar, non-synchronization trace interval unions with CSV runtime (sum durations for cycles); exclude synchronization-scope differences and rounding items below 0.1 us')


def pc(value):
    try:
        return hex(int(str(value),0))
    except (TypeError,ValueError):
        raise ValueError('Invalid instruction address: '+str(value)) from None


def number(value, integer=False):
    try:
        result=int(str(value),0) if integer else float(value)
    except (TypeError,ValueError):
        return None
    return result if math.isfinite(result) and result>=0 else None


def parse(directory, kernel, source, digest, case):
    """Read only the selected kernel directory; bind original per-core files without mixing cores."""
    roots=list(Path(directory).glob('OPPROF_*/simulator'))
    if len(roots)!=1:
        raise ValueError('Simulator report directory is missing or not unique')
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
            raise ValueError('Simulator instruction CSV is empty: '+str(path))
        cores.append(dict(name=path.parent.name,instructions=rows,listed_instruction_count=len(rows),
                          mapped_instruction_count=sum('source' in r for r in rows)))
    if not cores:
        raise ValueError('No per-core instruction CSV')
    return dict(profiler='msprof-op-simulator',source_sha256=digest,case=case,
                parser='hgkda public simulator CSV/trace adapter',kernels=[dict(name=kernel,cores=cores)])
