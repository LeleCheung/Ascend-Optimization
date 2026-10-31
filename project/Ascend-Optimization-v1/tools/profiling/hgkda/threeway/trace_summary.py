"""解析昇腾硬件任务；排除CPU事件及Overlap Analysis重复轨道。"""
from decimal import Decimal
import hashlib
import json
from pathlib import Path


def summarize(trace):
    trace=Path(trace)
    value=json.loads(trace.read_text(encoding='utf8'))
    events=value.get('traceEvents',[]) if isinstance(value,dict) else value
    hardware={event['pid'] for event in events if event.get('ph')=='M'
              and event.get('name')=='process_name'
              and event.get('args',{}).get('name')=='Ascend Hardware'}
    rows=[event for event in events if event.get('ph')=='X' and 'ts' in event and 'dur' in event
          and (event.get('pid') in hardware
               or str(event.get('args',{}).get('Task Type','')).startswith(('KERNEL_','MEMCPY','MEMSET'))
               or 'kernel' in str(event.get('cat','')).lower())]
    intervals=sorted((Decimal(str(event['ts'])),Decimal(str(event['ts']))+Decimal(str(event['dur']))) for event in rows)
    merged=[]
    for start,end in intervals:
        if end<start:raise ValueError('负设备任务耗时')
        if merged and start<=merged[-1][1]:merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
        else:merged.append((start,end))
    return dict(schema='ascend-device-path-v2',status='observed' if rows else 'no_device_tasks',
                device_tasks=rows,kernel_names=list(dict.fromkeys(event.get('name') for event in rows)),
                device_interval_union_us=float(sum((end-start for start,end in merged),Decimal(0))),
                device_elapsed_span_us=float(max(end for _,end in intervals)-intervals[0][0]) if intervals else None,
                trace=str(trace),trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                parser_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                selection='Ascend Hardware进程、明确Task Type或kernel类别；不包含Overlap Analysis聚合轨道；绝对时间以Decimal求区间',
                scope='单次独立profiling窗口的设备任务区间并集/跨度，不替代正式KGS完整调用')
