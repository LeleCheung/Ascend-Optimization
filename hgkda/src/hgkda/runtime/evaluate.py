"""Evaluate one reviewed candidate through an explicit KGS service and frozen checks."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from hgkda.adapters.kgs import evaluate, upload_bundle
from hgkda.feedback import performance_feedback
from hgkda.gate import admit
from hgkda.validation import supplement_status

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def check(config, kind, spec, source, output):
    """Run the delivered checker, never a repository-internal helper."""
    for name,digest in spec.get('contract_sha256',{}).items():
        assert sha(Path(spec['leaf'])/name)==digest,'contract changed'
    assert sha(spec['checker'])==spec['checker_sha256'],'checker changed'
    env=dict(config.get('device_environment',{}),ASCEND_RT_VISIBLE_DEVICES=str(config['device']),
             PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',TRITON_CACHE_DIR=str(output.parent/'checker-cache'))
    argv=[config['device_python'],spec['checker'],'--source',str(source),'--output',str(output)]
    if not spec.get('special'):argv+=['--operator',config['operator']]
    if spec.get('leaf'):argv+=['--leaf',spec['leaf']]
    shell='source '+shlex.quote(str(Path(config['sdk'])/'set_env.sh'))+'\n'
    shell+='\n'.join('export '+k+'='+shlex.quote(str(v)) for k,v in env.items())+'\nexec '+shlex.join(argv)
    command=['docker','exec','-w',str(output.parent),config['container'],'bash','-c',shell]
    with output.with_suffix('.log').open('xb') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    result=read(output);digest=sha(source)
    assert result['total']==spec['previous_total'] and len(result['cases'])==result['total']
    assert result['passed']==sum(bool(r['passed']) for r in result['cases'])
    if spec.get('special'):
        status='passed' if result['source_sha256']==digest and result['passed']==result['total'] else 'failed'
    else:status=supplement_status(config['operator'],result,digest)
    return dict(status=status,passed=result['passed'],total=result['total'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['config','source','review']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--port',type=int,help='Compatibility option; kgs_server must be configured explicitly')
    a=p.parse_args();c=read(a.config);server=c['kgs_server'];job=c['job'];digest=sha(a.source)
    admit(a.source.read_bytes().decode('utf8'),read(a.review))
    run=Path(c['run_directory'])/('formal-'+digest[:12]);run.mkdir(parents=True,exist_ok=False)
    source=run/'candidate.py';source.write_bytes(a.source.read_bytes())
    request=copy.deepcopy(read(job['request']));request['implementation']['sources'][0]['content']=source.read_bytes().decode('utf8')
    request['implementation']['name']+='_hgkda_'+digest[:12]
    save(run/'request.json',request);save(run/'review.json',read(a.review))
    archive=Path(job['archive']);assert 'sha256:'+sha(archive)==request['binding']['bundle_id']
    import fcntl
    with Path(c['lock']).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        checks={kind:check(c,kind,spec,source,run/(kind+'.json')) for kind,spec in job['checks'].items()}
        checks.update({kind:check(c,kind,spec,source,run/(kind+'.json')) for kind,spec in c.get('special_checks',{}).items()})
        assert all(row['status']=='passed' for row in checks.values()),'supplementary checks failed'
        upload_bundle(server,archive)
        value=evaluate(server,run/'request.json',run/'review.json',run/'evaluation')
    result=value['evaluation'];leaf=Path(job['checks']['cann_supplementary']['leaf'])
    timing=[json.loads(s) for s in (leaf/'timing.jsonl').read_text().splitlines() if s.strip()]
    expected={(json.loads(s)['name'],phase) for file,phase in [('correctness.jsonl','correctness'),('timing.jsonl','timing')]
              for s in (leaf/file).read_text().splitlines() if s.strip()}
    actual=[(x['uuid'],x['phase']) for x in result.get('per_workload',[])]
    assert len(actual)==len(expected) and set(actual)==expected,'complete case set differs'
    feedback=performance_feedback(result,timing,threshold=c.get('threshold',.8),checks_passed=True)
    feedback.update(source_sha256=digest,operator=c['operator'])
    save(run/'performance-feedback.json',dict(schema_version='hgkda.result/v1',status='completed',result=feedback))
    save(run/'audit-final.json',dict(evidence_verified=True,rows=[dict(operator=c['operator'],source_sha256=digest,
         full_passed=result['status']=='PASSED' and result['num_passed']==result['num_workloads'],
         formal_cases=result['num_workloads'],geo_mean=result.get('geo_mean'),accepted=feedback['accepted'],checks=checks)]))
    save(Path(c['run_directory'])/'workspace/performance-feedback.json',feedback)
    print(json.dumps(dict(status='completed',directory=str(run),decision=feedback['decision'],score=result.get('geo_mean'))))

if __name__=='__main__':main()
