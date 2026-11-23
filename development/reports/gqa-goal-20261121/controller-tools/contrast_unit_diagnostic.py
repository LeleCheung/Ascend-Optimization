"""Report raw per-PC clock comparisons without changing qualification rules."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
listing = json.loads((root / 'listing.json').read_bytes())
for core in listing['kernels'][0]['cores']:
    path = next(root.glob('raw/**/' + core['name'] + '/trace.json'))
    trace = json.loads(path.read_bytes())
    events = trace.get('traceEvents', []) if isinstance(trace, dict) else trace
    names = {e.get('pid'): e.get('args', {}).get('name') for e in events if e.get('ph') == 'M'}
    durations = {}
    counts = {}
    intervals = {}
    for e in events:
        args = e.get('args', {})
        if e.get('ph') != 'X' or 'pc_addr' not in args or names.get(e.get('pid')) == 'CACHEMISS':
            continue
        key = hex(int(str(args['pc_addr']), 0))
        durations[key] = durations.get(key, 0) + float(e.get('dur', 0))
        counts[key] = counts.get(key, 0) + 1
        start = float(e.get('ts', 0))
        intervals.setdefault(key, []).append((start, start + float(e.get('dur', 0))))
    rows = []
    for row in core['instructions']:
        ref = row.get('running_time_us')
        if not ref or ref < .1 or row['pc'] not in durations:
            continue
        if row.get('pipeline') in ['SCALAR', 'SCALARLDST', 'FLOWCTRL']:
            continue
        merged = []
        for start, end in sorted(intervals[row['pc']]):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(end, merged[-1][1])
            else:
                merged.append([start, end])
        union = sum(end-start for start, end in merged)
        rows.append(dict(pc=row['pc'], instruction=row['instruction'], pipe=row['pipeline'], union_us=union, union_ratio=union/ref,
                         csv_us=ref, trace_duration=durations[row['pc']], ratio=durations[row['pc']]/ref,
                         csv_calls=row.get('calls'), trace_calls=counts[row['pc']], cycles=row.get('cycles')))
    print(json.dumps(dict(core=core['name'], rows=rows), indent=2))
