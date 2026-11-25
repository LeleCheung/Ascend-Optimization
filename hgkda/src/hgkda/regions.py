"""Aggregate instructions by source regions; derive per-pipeline interval unions and observable idle time from timelines."""


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
    if not finite(start) or not finite(end) or end<=start:raise ValueError('Invalid capture range')
    active=union((max(start,a),min(end,b)) for a,b in intervals)
    gaps=[];cursor=start
    for a,b in active:
        if a>cursor:gaps.append([cursor,a])
        cursor=max(cursor,b)
    if cursor<end:gaps.append([cursor,end])
    return gaps


def subtract(intervals, occupied):
    # Merge once, then sweep. Re-sorting every occupied interval for each wait
    # makes long simulator traces quadratic in the number of instructions.
    result=[];busy=union(occupied);j=0
    for a,b in union(intervals):
        while j<len(busy) and busy[j][1]<=a:j+=1
        cursor=a;k=j
        while k<len(busy) and busy[k][0]<b:
            start,end=busy[k]
            if start>cursor:result.append([cursor,min(start,b)])
            cursor=max(cursor,end)
            if cursor>=b:break
            k+=1
        if cursor<b:result.append([cursor,b])
        j=k
    return result


def pc_key(value):
    try:return int(value,16) if isinstance(value,str) else value if isinstance(value,int) else None
    except ValueError:return None


def source_scopes(sources):
    """Identify static loop, conditional, and function ranges using SHA-verified source with exact filename matching."""
    scopes={}
    for file,entry in (sources or {}).items():
        text=entry['text']
        if hashlib.sha256(text.encode('utf8')).hexdigest()!=entry['sha256']:
            raise ValueError('Hotspot-region source SHA mismatch: '+file)
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
        # Keep loads, computation, and stores within a loop together rather than splitting the loop again at each conditional.
        for kind in ['source_loop','source_conditional','source_function']:
            matches=[s for s in spans if s[0]==kind]
            if matches:return min(matches,key=lambda s:s[3]-s[2])
        start=((line-1)//width)*width+1
        return ('source_neighborhood',source['file'],start,start+width-1)
    try:pc=int(instruction['pc'],16)
    except (KeyError,ValueError,TypeError):return ('unmapped','unknown',0,0)
    return ('pc_neighborhood','unmapped',pc//64*64,pc//64*64+63)


def analyze(listing, trace=None, *, core=None, kernel=None, width=8, top=12, trace_unit='us', capture=None, sources=None):
    if width<=0 or top<=0 or trace_unit not in {'us','ns','cycles','fraction'}:raise ValueError('Invalid grouping or time unit')
    kernels=[k for k in listing.get('kernels',[]) if kernel is None or k['name']==kernel]
    if not kernels:raise ValueError('No matching kernel')
    if trace is not None and (len(kernels)!=1 or core is None):raise ValueError('Trace must bind to one kernel and a specific hardware core')
    if trace is not None and isinstance(trace,dict) and 'hgkda_binding' in trace:
        binding=trace['hgkda_binding']
        expected=dict(source_sha256=listing.get('source_sha256'),case=listing.get('case'),
                      kernel=kernels[0]['name'],core=core,time_unit=trace_unit)
        if any(binding.get(key)!=value for key,value in expected.items()):
            raise ValueError('Timeline and instruction listing disagree on source, case, kernel, core, or time unit')
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
                         wait_instruction_time_sum_us=sum(m.get('running_time_us',0) for m in members if re.search(r'WAIT|BAR',m.get('instruction',''),re.I) and finite(m.get('running_time_us',0))),
                         pcs=[m.get('pc') for m in members],source_lines=sorted({(m.get('source') or {}).get('line',0) for m in members}),
                         contains_function_entry=any((m.get('source') or {}).get('code','').lstrip().startswith('def ') for m in members),
                         interpretation=('Static AST ranges from SHA-verified source; compiler debug locations determine membership, not dynamic iteration counts or hardware critical paths. ' if kind in {'source_loop','source_conditional','source_function'} else 'Source/PC neighborhood. ')+'Cumulative instruction time ranks regions for investigation; overlapping pipelines prevent interpreting it as elapsed region time or established causality')
                regions.append(row)
                for m in members:
                    pc=pc_key(m.get('pc'))
                    if pc is not None:lookup[pc]=identifier
    timeline=dict(status='unsupported',reason='No bound instruction timeline; aggregate activity ratios cannot reconstruct idle intervals')
    if trace is not None:
        events=trace.get('traceEvents',[]) if isinstance(trace,dict) else trace
        names={e.get('pid'):e.get('args',{}).get('name','unknown') for e in events if e.get('ph')=='M' and e.get('name')=='process_name'}
        valid=[e for e in events if e.get('ph')=='X' and finite(e.get('ts')) and finite(e.get('dur')) and e['dur']>0]
        hardware=[e for e in valid if names.get(e.get('pid')) in HARDWARE_PIPELINES]
        if not valid:
            timeline=dict(status='unsupported',reason='No complete events with positive duration')
        else:
            window_events=hardware or valid
            start,end=capture if capture is not None else (min(e['ts'] for e in window_events),max(e['ts']+e['dur'] for e in window_events))
            if not finite(start) or not finite(end) or end<=start:raise ValueError('Invalid timeline capture window')
            by_pipeline=defaultdict(list);waits=defaultdict(list);productive=defaultdict(list);by_region=defaultdict(list)
            outside=0
            for e in valid:
                a,b=max(start,e['ts']),min(end,e['ts']+e['dur'])
                if b<=a:outside+=1;continue
                pipe=names.get(e.get('pid'),str(e.get('pid')))
                by_pipeline[pipe].append((a,b))
                if re.search(r'WAIT|BAR',e.get('name',''),re.I):waits[pipe].append((a,b))
                else:productive[pipe].append((a,b))
                pc=pc_key(e.get('args',{}).get('pc_addr'))
                if pc in lookup and pipe in HARDWARE_PIPELINES:by_region[lookup[pc]].append((a,b))
            rows=[]
            # Declared hardware lanes with no events still have observable empty intervals.
            for pipe in set(names.values()) & HARDWARE_PIPELINES:
                by_pipeline.setdefault(pipe,[])
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
                          capture_scope='Explicit window' if capture else 'First through last visible hardware-pipeline event on this core; use all events and label the fallback when hardware lanes are absent. Unknown idle time outside the window is excluded.',
                          display_time_unit=trace.get('displayTimeUnit') if isinstance(trace,dict) else None,
                          pipelines=rows,outside_events=outside,
                          interpretation='productive excludes WAIT/BAR; wait_only excludes waits overlapping productive instructions; productive+wait_only+idle equals the window. Idle does not establish a causal stall or critical path. Auxiliary trace lanes are labeled separately. Chrome ts/dur usually uses microseconds; displayTimeUnit is display-only, and callers must confirm the input unit.')
    regions.sort(key=lambda r:(r.get('trace_interval_union',0),r['instruction_time_sum_us']),reverse=True)
    return dict(schema='hgkda-regions-v1',status='completed' if regions else 'empty',
                source_sha256=listing.get('source_sha256'),case=listing.get('case'),
                regions=regions[:top],total_regions=len(regions),timeline=timeline,
                grouping=dict(source_line_window=width,pc_bytes=64,top=top,verified_source_files={file:entry['sha256'] for file,entry in (sources or {}).items()},
                    policy='Prefer static source loops, then conditional/function ranges with exact file matching; otherwise retain neighborhood aggregation'),
                limitations=['Static source ranges are not compiled basic blocks or dynamic critical paths; interpret neighborhood fallbacks alongside actual IR', 'Region intervals may overlap and cannot be summed into total latency', 'Function-entry debug lines may cover several instruction regions and are explicitly labeled'])
