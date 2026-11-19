from pathlib import Path
import sys,json,hashlib
r=Path('/data/hanle/ascend-optimization/runs/gqa-attribution-20261117')
sys.path.insert(0,str(r/'method/hgkda/src'))
from hgkda.simulator import parse,validate_time_unit
from hgkda.regions import analyze
after='--after' in sys.argv
out=r/('capture-after/simulator' if after else 'capture/simulator')
source=(r/'with-simulator-v2/evaluation-round-02/source.py') if after else Path('/data/hanle/ascend-optimization/runs/night-research-20261116/gqa/workspace/candidate.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
case=next(x for x in json.loads((r/'contract.json').read_bytes())['cases'] if x['case_id']==4)
kernel='_gqa_fwd_kernel_mix_aic'
listing=parse(out/'raw',kernel,source,sha(source),case);save(out/'listing.json',listing)
print('kernels',[(k['name'],len(k['cores'])) for k in listing['kernels']])
reports=[]
for core in listing['kernels'][0]['cores']:
    trace_file=next((out/'raw').glob('OPPROF_*/simulator/'+core['name']+'/trace.json'))
    raw=json.loads(trace_file.read_bytes());trace=dict(raw) if isinstance(raw,dict) else dict(traceEvents=raw)
    unit=validate_time_unit(raw,core['instructions'],'us')
    trace['hgkda_binding']=dict(source_sha256=sha(source),case=case,kernel=kernel,core=core['name'],time_unit='us',raw_trace_sha256=sha(trace_file),raw_trace=str(trace_file),unit_validation=unit,scope='partial observed simulator window; application timed out; not full invocation')
    result=analyze(listing,trace,kernel=kernel,core=core['name'],trace_unit='us',sources={str(source):dict(text=source.read_text(),sha256=sha(source))})
    result['collection_qualification']={'status':'partial_timeout','kernel_completed':False,'reason':'Application timed out; no post-synchronize completion manifest. Only observed instruction/window facts may be used.'}
    save(out/(core['name']+'-regions.json'),result)
    reports.append({'core':core['name'],'unit_validation':unit,'timeline':result['timeline'],'regions':result.get('regions'),
     'top_instructions':core['instructions'][:12],'qualification':result['collection_qualification']})
save(out/'summary.json',{'source_sha256':sha(source),'case':case,'kernel':kernel,'reports':reports,
 'scope':'partial simulator capture; no complete-invocation or critical-path claim; instruction time may overlap'})
print('reports',len(reports))
