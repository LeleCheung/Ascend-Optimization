"""Preserve useful instruction evidence when independent trace-clock checks fail."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--directory',type=Path,required=True)
a=p.parse_args();r=a.run;out=a.directory;c=json.loads((r/'config.json').read_bytes())
sys.path.insert(0,str(Path(c['package'])/'src'))
from hgkda.regions import analyze
from hgkda.runtime.transfer_context import compact_regions
spec=json.loads((out/'spec.json').read_bytes());listing=json.loads((out/'listing.json').read_bytes())
source=Path(spec['source']);digest=hashlib.sha256(source.read_bytes()).hexdigest();assert digest==spec['source_sha256']
request=json.loads((out/'request.json').read_bytes());kernel=request['kernel']
assert request['source_sha256']==digest and listing['kernels'][0]['name']==kernel
reports=[]
for core in listing['kernels'][0]['cores']:
    instructions=core['instructions'];assert instructions
    result=analyze(listing,kernel=kernel,core=core['name'],sources={str(source):dict(text=source.read_text(),sha256=digest)})
    counts=Counter()
    for item in instructions:counts[item['pipeline']]+=item.get('calls') or 0
    reports.append(dict(core=core['name'],**compact_regions(result),instruction_calls_by_pipeline=dict(counts),
        top_instructions=sorted(instructions,key=lambda x:x.get('running_time_us') or 0,reverse=True)[:16]))
summary=dict(status='qualified_observation',source_sha256=digest,case=spec['case'],kernel=kernel,reports=reports,
    capture_scope='Partial instruction CSV observation only; application timed out and trace time unit could not be independently validated.',
    application_completed=False,timeline_qualified=False,
    limitations=['Use instruction kinds, call counts, source mappings and CSV-reported ranking only.',
        'Do not derive timeline execution/wait/idle ratios, critical paths or complete invocation time from this observation.'],
    failed_timeline_summary_log=str(out/'summary.log'))
(out/'instruction-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(output=str(out/'instruction-summary.json'),counts=[x['instruction_calls_by_pipeline'] for x in reports])))
