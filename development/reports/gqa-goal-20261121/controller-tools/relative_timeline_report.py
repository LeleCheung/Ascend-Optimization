"""Analyze existing raw simulator traces in relative-window units, preserving originals."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--package',type=Path,required=True)
a=p.parse_args();out=a.directory;sys.path.insert(0,str(a.package/'src'))
from hgkda.timeline import relative_timeline
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
listing=json.loads((out/'listing.json').read_bytes());spec=json.loads((out/'spec.json').read_bytes())
source=Path(spec['source']);assert sha(source)==spec['source_sha256']==listing['source_sha256']
reports=[]
for kernel in listing['kernels']:
    for core in kernel['cores']:
        paths=list((out/'raw').glob('**/'+core['name']+'/trace.json'));assert len(paths)==1
        raw=paths[0].read_bytes()
        binding=dict(source_sha256=spec['source_sha256'],case=spec['case'],kernel=kernel['name'],core=core['name'],raw_trace_sha256=sha(paths[0]))
        report=relative_timeline(listing,raw,binding=binding,sources={str(source):dict(text=source.read_text(),sha256=sha(source))})
        dest=out/(core['name']+'-relative-timeline.json');dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        reports.append(dict(core=core['name'],report=str(dest),timeline=report['timeline']))
result=dict(status='completed_relative_analysis',source_sha256=spec['source_sha256'],case=spec['case'],reports=reports,
    analyzer_sha256=sha(a.package/'src/hgkda/timeline.py'),regions_sha256=sha(a.package/'src/hgkda/regions.py'),
    absolute_time_qualified=False,scope='Retrospective analysis of preserved partial trace; no new device capture or absolute clock claim.')
(out/'relative-timeline-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps([dict(core=x['core'],pipelines=[{k:row[k] for k in ['pipeline','productive_ratio','wait_only_ratio','idle_ratio']}
    for row in x['timeline']['pipelines'] if row['pipeline_kind']=='hardware_execution']) for x in reports]))
