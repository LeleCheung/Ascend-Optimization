"""Relative hardware-pipeline occupancy from a source-bound common-clock trace."""
import copy
import hashlib
import json

from .regions import HARDWARE_PIPELINES, analyze, finite


def relative_timeline(listing, raw_trace, *, binding, sources=None):
    """Normalize one hardware-core window without assigning an absolute time unit.

    ts and dur share the vendor trace's common clock. A positive uniform clock
    scale cancels in occupancy ratios; CSV timestamp agreement is not claimed.
    """
    digest=hashlib.sha256(raw_trace).hexdigest()
    if binding.get('raw_trace_sha256')!=digest:
        raise ValueError('Raw trace hash differs from binding')
    if any(binding.get(k)!=listing.get(k) for k in ['source_sha256','case']):
        raise ValueError('Trace source or case differs from instruction listing')
    kernels=[k for k in listing.get('kernels',[]) if k['name']==binding.get('kernel')]
    if len(kernels)!=1 or binding.get('core') not in [c['name'] for c in kernels[0]['cores']]:
        raise ValueError('Trace needs an existing uniquely selected kernel/core')
    raw=json.loads(raw_trace)
    events=raw.get('traceEvents',[]) if isinstance(raw,dict) else raw
    names={e.get('pid'):e.get('args',{}).get('name') for e in events
           if e.get('ph')=='M' and e.get('name')=='process_name'}
    hardware=[e for e in events if e.get('ph')=='X' and names.get(e.get('pid')) in HARDWARE_PIPELINES
              and finite(e.get('ts')) and finite(e.get('dur')) and e['dur']>0]
    if not hardware:raise ValueError('No positive-duration hardware events')
    start=min(e['ts'] for e in hardware);end=max(e['ts']+e['dur'] for e in hardware)
    span=end-start
    if not finite(span) or span<=0:raise ValueError('Invalid hardware capture window')
    normalized=[]
    for event in events:
        event=copy.deepcopy(event)
        if event.get('ph')=='X':
            if not finite(event.get('ts')) or not finite(event.get('dur')) or event['dur']<=0:continue
            event['ts']=(event['ts']-start)/span;event['dur']/=span
        normalized.append(event)
    trace=dict(traceEvents=normalized,hgkda_binding=dict(binding,time_unit='fraction'))
    result=analyze(listing,trace,kernel=binding['kernel'],core=binding['core'],
                   trace_unit='fraction',capture=(0.,1.),sources=sources)
    timeline=result['timeline']
    for row in timeline['pipelines']:
        partition=row['productive_ratio']+row['wait_only_ratio']+row['idle_ratio']
        if abs(partition-1)>1e-8:raise ValueError('Pipeline partition does not cover normalized window')
    timeline.update(clock_qualification='relative_only',absolute_time_qualified=False,
        raw_trace_sha256=digest,raw_clock_window=dict(start=start,end=end,unit='unassigned'),
        display_time_unit='window_fraction',capture_scope='First to last visible hardware event on this core; partial capture boundaries preserved.',
        interpretation='Occupancy fractions on a common trace clock; explicit WAIT/BAR only. Empty intervals describe the observed window. No full-kernel duration or dependency-stall attribution.')
    result.setdefault('limitations',[]).append('Absolute clock unit is unqualified; use normalized fractions only. Hardware lanes omitted from trace metadata remain unknown.')
    return result
