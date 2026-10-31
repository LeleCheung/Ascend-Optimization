"""指令清单按源码邻域聚合；有时间轴时计算逐流水区间并集与可观测空闲。"""
import argparse
import ast
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

HARDWARE_PIPELINES={'VECTOR','CUBE','MTE1','MTE2','MTE3','FIXP','SCALAR','SCALARLDST'}


def finite(x):
    return isinstance(x, (int,float)) and not isinstance(x,bool) and math.isfinite(x)


def union(intervals):
    rows=sorted((float(a),float(b)) for a,b in intervals if finite(a) and finite(b) and b>a)
    merged=[]
    for a,b in rows:
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    return merged


def length(intervals):
    return sum(b-a for a,b in union(intervals))


def complement(intervals, start, end):
    if not finite(start) or not finite(end) or end<=start:raise ValueError('无效捕获范围')
    active=union((max(start,a),min(end,b)) for a,b in intervals)
    gaps=[];cursor=start
    for a,b in active:
        if a>cursor:gaps.append([cursor,a])
        cursor=max(cursor,b)
    if cursor<end:gaps.append([cursor,end])
    return gaps


def subtract(intervals, occupied):
    result=[]
    for a,b in union(intervals):result.extend(complement(occupied,a,b))
    return result


def pc_key(value):
    try:return int(value,16) if isinstance(value,str) else value if isinstance(value,int) else None
    except ValueError:return None


def source_scopes(sources):
    """只使用SHA核验且文件名精确匹配的源码，识别静态循环/条件/函数范围。"""
    scopes={}
    for file,entry in (sources or {}).items():
        text=entry['text']
        if hashlib.sha256(text.encode('utf8')).hexdigest()!=entry['sha256']:
            raise ValueError('热点区域源码SHA不匹配 '+file)
        spans=[]
        for node in ast.walk(ast.parse(text,filename=file)):
            if isinstance(node,(ast.For,ast.While)):
                kind='source_loop'
            elif isinstance(node,ast.If):kind='source_conditional'
            elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):kind='source_function'
            else:continue
            spans.append((kind,file,node.lineno,node.end_lineno))
        scopes[file]=spans
    return scopes


def region_key(instruction, width=8, scopes=None):
    source=instruction.get('source') or {}
    line=source.get('line')
    if isinstance(line,int) and line>0 and source.get('file'):
        spans=[s for s in (scopes or {}).get(source['file'],[]) if s[2]<=line<=s[3]]
        # 优先把循环中的搬入/计算/写回放在同一范围，避免按if分支再次割碎循环。
        for kind in ['source_loop','source_conditional','source_function']:
            matches=[s for s in spans if s[0]==kind]
            if matches:return min(matches,key=lambda s:s[3]-s[2])
        start=((line-1)//width)*width+1
        return ('source_neighborhood',source['file'],start,start+width-1)
    try:pc=int(instruction['pc'],16)
    except (KeyError,ValueError,TypeError):return ('unmapped','unknown',0,0)
    return ('pc_neighborhood','unmapped',pc//64*64,pc//64*64+63)


def analyze(listing, trace=None, *, core=None, kernel=None, width=8, top=12, trace_unit='us', capture=None, sources=None):
    if width<=0 or top<=0 or trace_unit not in {'us','ns','cycles'}:raise ValueError('无效分组或单位')
    kernels=[k for k in listing.get('kernels',[]) if kernel is None or k['name']==kernel]
    if not kernels:raise ValueError('没有匹配kernel')
    if trace is not None and (len(kernels)!=1 or core is None):raise ValueError('trace必须绑定唯一kernel和具体硬件核')
    regions=[];lookup={};scopes=source_scopes(sources)
    for k in kernels:
        for c in k.get('cores',[]):
            if core is not None and c['name']!=core:continue
            groups=defaultdict(list)
            for instruction in c.get('instructions',[]):groups[region_key(instruction,width,scopes)].append(instruction)
            for key, members in groups.items():
                kind,file,start,end=key
                identifier=k['name']+'|'+c['name']+'|'+str(key)
                times=[m.get('running_time_us') for m in members if finite(m.get('running_time_us'))]
                row=dict(id=identifier,kernel=k['name'],core=c['name'],kind=kind,file=file,start=start,end=end,
                         instructions=len(members),pipelines=sorted({str(m.get('pipeline','unknown')) for m in members}),
                         instruction_time_sum_us=sum(times),largest_instruction_us=max(times,default=0),
                         wait_instruction_time_sum_us=sum(m.get('running_time_us',0) for m in members if re.search(r'WAIT|BAR',m.get('instruction','')) and finite(m.get('running_time_us',0))),
                         pcs=[m.get('pc') for m in members],source_lines=sorted({(m.get('source') or {}).get('line',0) for m in members}),
                         contains_function_entry=any((m.get('source') or {}).get('code','').lstrip().startswith('def ') for m in members),
                         interpretation=('SHA核验源码的静态AST范围；编译器调试位置决定归属，不重建动态循环次数或硬件关键路径。' if kind in {'source_loop','source_conditional','source_function'} else '源码邻域/PC邻域。')+'累计指令时间仅作关注度排序，各流水可重叠，非真实区域耗时或已证实因果')
                regions.append(row)
                for m in members:
                    pc=pc_key(m.get('pc'))
                    if pc is not None:lookup[pc]=identifier
    timeline=dict(status='unsupported',reason='没有绑定的逐指令时间轴；不从汇总活跃比例反推空泡')
    if trace is not None:
        events=trace.get('traceEvents',[]) if isinstance(trace,dict) else trace
        names={e.get('pid'):e.get('args',{}).get('name','unknown') for e in events if e.get('ph')=='M' and e.get('name')=='process_name'}
        valid=[e for e in events if e.get('ph')=='X' and finite(e.get('ts')) and finite(e.get('dur')) and e['dur']>0]
        hardware=[e for e in valid if names.get(e.get('pid')) in HARDWARE_PIPELINES]
        if not valid:
            timeline=dict(status='unsupported',reason='无正时长完整事件')
        else:
            window_events=hardware or valid
            start,end=capture if capture is not None else (min(e['ts'] for e in window_events),max(e['ts']+e['dur'] for e in window_events))
            if not finite(start) or not finite(end) or end<=start:raise ValueError('无效时间轴捕获窗口')
            by_pipeline=defaultdict(list);waits=defaultdict(list);productive=defaultdict(list);by_region=defaultdict(list)
            outside=0
            for e in valid:
                a,b=max(start,e['ts']),min(end,e['ts']+e['dur'])
                if b<=a:outside+=1;continue
                pipe=names.get(e.get('pid'),str(e.get('pid')))
                by_pipeline[pipe].append((a,b))
                if re.search(r'WAIT|BAR',e.get('name','')):waits[pipe].append((a,b))
                else:productive[pipe].append((a,b))
                pc=pc_key(e.get('args',{}).get('pc_addr'))
                if pc in lookup and pipe in HARDWARE_PIPELINES:by_region[lookup[pc]].append((a,b))
            rows=[]
            for pipe,intervals in sorted(by_pipeline.items()):
                active=union(intervals);gaps=complement(active,start,end);wait=union(waits[pipe])
                productive_time=length(productive[pipe]);wait_only=length(subtract(wait,productive[pipe]))
                rows.append(dict(pipeline=pipe,event_count=len(intervals),active_union=length(active),active_ratio=length(active)/(end-start),
                                 wait_union=length(wait),wait_ratio=length(wait)/(end-start),idle_union=length(gaps),idle_ratio=length(gaps)/(end-start),
                                 productive_union=productive_time,productive_ratio=productive_time/(end-start),wait_only_union=wait_only,wait_only_ratio=wait_only/(end-start),
                                 pipeline_kind='hardware_execution' if pipe in HARDWARE_PIPELINES else 'auxiliary_trace_lane',
                                 largest_idle=max((b-a for a,b in gaps),default=0),idle_intervals=gaps[:64],idle_intervals_total=len(gaps),
                                 activity_includes_wait=bool(wait)))
            for row in regions:
                if row['id'] in by_region:
                    row.update(trace_interval_union=length(by_region[row['id']]),trace_time_unit=trace_unit)
            timeline=dict(status='completed',core=core,kernel=kernels[0]['name'],time_unit=trace_unit,capture_start=start,capture_end=end,capture_duration=end-start,
                          capture_scope='显式窗口' if capture else '本核硬件流水首个至末个可见事件；无硬件泳道时使用全部事件并单列；不包含未知启动前后空闲',
                          display_time_unit=trace.get('displayTimeUnit') if isinstance(trace,dict) else None,
                          pipelines=rows,outside_events=outside,
                          interpretation='productive排除WAIT/BAR；wait_only排除与有效指令重叠的等待；productive+wait_only+idle=窗口。idle不是因果停顿，不重建关键路径；辅助trace泳道另标。Chrome ts/dur通常为微秒，displayTimeUnit仅显示单位，输入单位由调用者确认。')
    regions.sort(key=lambda r:(r.get('trace_interval_union',0),r['instruction_time_sum_us']),reverse=True)
    return dict(schema='hgkda-regions-v1',status='completed' if regions else 'empty',regions=regions[:top],total_regions=len(regions),timeline=timeline,
                grouping=dict(source_line_window=width,pc_bytes=64,top=top,verified_source_files={file:entry['sha256'] for file,entry in (sources or {}).items()},
                    policy='精确文件匹配时优先静态源码循环，其次条件/函数范围；无核验源码时保留邻域聚合'),
                limitations=['静态源码范围不等同编译后基本块或动态关键路径；源码邻域回退仍需结合实际IR解释', '不同区域的区间也可能重叠，不累加为总耗时', '函数入口调试行号可能覆盖多段指令，已显式标记'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--listing',type=Path,required=True);p.add_argument('--trace',type=Path);p.add_argument('--core');p.add_argument('--kernel');p.add_argument('--output',type=Path,required=True);p.add_argument('--trace-unit',choices=['us','ns','cycles'],default='us');p.add_argument('--window',type=int,default=8)
    p.add_argument('--source-map',type=Path,help='JSON映射：调试信息文件名 -> {path: 冻结源码路径, sha256: 预期SHA}')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    sources={}
    if a.source_map:
        for file,entry in json.loads(a.source_map.read_text(encoding='utf8')).items():
            raw=Path(entry['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('冻结源码SHA不同 '+entry['path'])
            sources[file]=dict(text=raw.decode('utf8'),sha256=entry['sha256'])
    result=analyze(json.loads(a.listing.read_text(encoding='utf8')),json.loads(a.trace.read_text(encoding='utf8')) if a.trace else None,core=a.core,kernel=a.kernel,width=a.window,trace_unit=a.trace_unit,sources=sources)
    result['inputs']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [a.listing,a.trace,a.source_map] if f}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=result['status'],regions=result['total_regions'],timeline=result['timeline']['status']),ensure_ascii=False))


if __name__=='__main__':main()
