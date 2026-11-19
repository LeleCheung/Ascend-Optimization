"""Summarize a source-bound simulator capture with explicit completion and launch identity."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--study',type=Path,required=True);a=p.parse_args()
    out=a.directory;r=a.study
    sys.path.insert(0,str(r/'method/hgkda/src'))
    from hgkda.simulator import parse,validate_time_unit
    from hgkda.runtime.transfer_context import compact_regions
    spec_path=out/'sim-spec.json' if (out/'sim-spec.json').exists() else out/'spec.json'
    spec=json.loads(spec_path.read_bytes());source=Path(spec['source'])
    sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
    assert sha(source)==spec['source_sha256']
    request=json.loads((out/'request.json').read_bytes())
    kernel=request.get('kernel') or next(c.split('=',1)[1] for c in request['command'] if c.startswith('--kernel-name='))
    raw_dir=out/'raw/kernel-000' if (out/'raw/kernel-000').exists() else out/'raw'
    listing=parse(raw_dir,kernel,source,sha(source),spec['case'])
    save=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    save(out/'listing.json',listing)
    modspec=importlib.util.spec_from_file_location('contrast_regions',r/'regions_optimized.py')
    analyzer=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(analyzer)
    reports=[]
    for core in listing['kernels'][0]['cores']:
        trace_file=next(raw_dir.glob('OPPROF_*/simulator/'+core['name']+'/trace.json'))
        raw=json.loads(trace_file.read_bytes());trace=dict(raw) if isinstance(raw,dict) else dict(traceEvents=raw)
        unit=validate_time_unit(raw,core['instructions'],'us')
        trace['hgkda_binding']=dict(source_sha256=sha(source),case=spec['case'],kernel=kernel,core=core['name'],time_unit='us',raw_trace_sha256=sha(trace_file),unit_validation=unit)
        result=analyzer.analyze(listing,trace,kernel=kernel,core=core['name'],trace_unit='us',sources={str(source):dict(text=source.read_text(),sha256=sha(source))})
        assert result['timeline']['status']=='completed'
        save(out/(core['name']+'-regions.json'),result)
        compact=compact_regions(result)
        reports.append(dict(core=core['name'],**compact,unit_validation=unit,top_instructions=sorted(core['instructions'],key=lambda x:x.get('running_time_us') or 0,reverse=True)[:12]))
    assert reports
    complete=(Path(spec['ir_directory'])/'manifest.json').exists()
    cache=Path(spec['autotune_replay']) if spec.get('autotune_replay') else None
    summary=dict(status='qualified_observation',source_sha256=sha(source),case=spec['case'],kernel=kernel,
        reports=reports,application_completed=complete,capture_scope='completed selected invocation' if complete else 'partial window; application did not finish',
        parser_sha256=sha(r/'regions_optimized.py'),limitations=['Instruction times overlap; no full-device critical-path or speedup claim.'])
    if cache:
        assert sha(cache)==request['autotune_cache_sha256']
        summary['replayed_configuration']=json.loads(cache.read_bytes())
        summary['configuration_identity']='real-device selected configuration injected into autotune cache before simulation; algorithm bytes unchanged'
    save(out/'summary.json',summary)
    print(json.dumps({'summary':str(out/'summary.json'),'completed':complete,'cores':len(reports)}))


if __name__=='__main__':main()
