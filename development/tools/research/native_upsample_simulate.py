"""Collect native baseline simulation without rewriting it to satisfy candidate admission."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import fcntl

r=Path('/data/hanle/ascend-optimization/runs/native-upsample-contrast-20261118')
run=r/'native';pair=run/'paired';out=run/'simulator'
sys.path.insert(0,str(r/'method/hgkda/src'))
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    spec=json.loads((pair/'spec.json').read_bytes())
    discovery=json.loads((pair/'candidate-discovery.json').read_bytes())
    assert len(discovery['kernels'])==1
    kernel=discovery['kernels'][0]
    out.mkdir(exist_ok=False)
    spec['ir_directory']=str(out/'actual-ir');save(out/'spec.json',spec)
    driver=r/'method/hgkda/src/hgkda/runtime/paired_driver.py'
    app=shlex.join(['/usr/local/python3.11.15/bin/python3.11',str(driver),'--spec',str(out/'spec.json'),'--side','candidate'])
    command=['msprof','op','simulator','--application='+app,'--kernel-name='+kernel,'--soc-version=Ascend910B4','--core-id=0','--launch-count=1','--dump=on','--timeout=5','--output='+str(out/'raw')]
    deps='/data/hanle/ascend-optimization/Ascend-Optimization/development/dependencies/third_party/FlagGems/src'
    shell='source /usr/local/Ascend/cann-9.0.0/set_env.sh\nexport ASCEND_RT_VISIBLE_DEVICES=7\nexport TRITON_DISABLE_LINE_INFO=false\nexport PYTHONPATH='+deps+'\nexport TRITON_CACHE_DIR='+str(out/'cache')+'\nexport LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/aarch64-linux/simulator/Ascend910B4/lib:"${LD_LIBRARY_PATH:-}"\nexec '+shlex.join(command)
    save(out/'request.json',dict(command=command,source_sha256=spec['source_sha256'],case=spec['case'],entrypoint=spec['entrypoint'],scope='native baseline, not generated candidate',driver_sha256=sha(driver)))
    with (r.parent/'physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with (out/'collector.log').open('w') as log:
            result=subprocess.run(['docker','exec','-w',str(out),'tle_yy','bash','-c',shell],stdout=log,stderr=subprocess.STDOUT,timeout=450)
    save(out/'execution.json',dict(returncode=result.returncode))
    from hgkda.simulator import parse,validate_time_unit
    from hgkda.regions import analyze
    source=Path(spec['source']);listing=parse(out/'raw',kernel,source,sha(source),spec['case']);save(out/'listing.json',listing)
    reports=[]
    complete=(out/'actual-ir/manifest.json').exists()
    for core in listing['kernels'][0]['cores']:
        paths=list((out/'raw').glob('OPPROF_*/simulator/'+core['name']+'/trace.json'));assert len(paths)==1
        raw=json.loads(paths[0].read_bytes());trace=dict(raw) if isinstance(raw,dict) else dict(traceEvents=raw)
        unit=validate_time_unit(raw,core['instructions'],'us')
        trace['hgkda_binding']=dict(source_sha256=sha(source),case=spec['case'],kernel=kernel,core=core['name'],time_unit='us',raw_trace_sha256=sha(paths[0]),unit_validation=unit)
        report=analyze(listing,trace,kernel=kernel,core=core['name'],trace_unit='us',sources={str(source):dict(text=source.read_text(),sha256=sha(source))})
        assert report['timeline']['status']=='completed'
        save(out/(core['name']+'-regions.json'),report)
        reports.append(dict(core=core['name'],timeline=report['timeline'],regions=report['regions'],limitations=report['limitations'],unit_validation=unit))
    assert reports
    save(out/'summary.json',dict(status='qualified_observation',source_sha256=sha(source),case=spec['case'],kernel=kernel,
        application_completed=complete,capture_scope='complete invocation' if complete else 'partial observed window; application did not complete',
        reports=reports,limitation='Observed instruction intervals are not a causal critical path or full-device runtime.'))
    print(json.dumps({'summary':str(out/'summary.json'),'application_completed':complete}))

if __name__=='__main__':main()
