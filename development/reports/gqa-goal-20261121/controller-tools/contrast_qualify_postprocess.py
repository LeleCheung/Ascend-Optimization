"""Recover complete metric tables when only msopprof report postprocessing remains."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--pid',type=int,required=True);p.add_argument('--study',type=Path,required=True)
    a=p.parse_args();out=a.directory.resolve();assert out.is_relative_to(a.study.resolve())
    sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
    save=lambda x,v:x.write_text(json.dumps(v,indent=2)+'\n')
    spec=json.loads((out/'spec.json').read_bytes());source=Path(spec['source'])
    assert sha(source)==spec['source_sha256']
    manifest=out/'actual-ir/manifest.json';m=json.loads(manifest.read_bytes())
    assert m['source_sha256']==spec['source_sha256']
    for f in m['files']:assert sha(manifest.parent/f['path'])==f['sha256']
    proc=Path('/proc')/str(a.pid)
    command=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert '/tools/msopprof/bin/msopprof ' in command
    assert '--spec '+str(out/'spec.json')+' --side candidate' in command
    # Every task must have no surviving child, including the launched application.
    def no_children():
        return all(not q.read_text().strip() for q in proc.glob('task/*/children'))
    assert no_children()
    basics=list((out/'candidate/profile').rglob('OpBasicInfo*.csv'));assert len(basics)==1
    with basics[0].open() as f:basic=list(csv.DictReader(f))
    assert len(basic)==1
    kernel=basic[0]['Op Name'];blocks=int(basic[0]['Block Dim'])
    assert float(basic[0]['Task Duration(us)'])>0
    discovery=out/'candidate-discovery.json';d=json.loads(discovery.read_bytes())
    assert d['source_sha256']==spec['source_sha256'] and d['kernels']==[kernel]
    categories=['OpBasicInfo','ArithmeticUtilization','PipeUtilization','Memory','MemoryUB','MemoryL0','L2Cache','ResourceConflictRatio']
    records=[]
    for category in categories:
        paths=list(basics[0].parent.glob(category+'_*.csv'));assert len(paths)==1
        table=paths[0]
        with table.open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==(1 if category=='OpBasicInfo' else blocks),(category,len(rows),blocks)
        records.append(dict(path=str(table),sha256=sha(table),rows=len(rows)))
    time.sleep(5)
    assert no_children() and proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()==command
    for record in records:assert sha(Path(record['path']))==record['sha256']
    q=dict(status='kernel_completed_report_postprocess_terminated',pid=a.pid,command=command,
           source_sha256=spec['source_sha256'],completion_manifest_sha256=sha(manifest),csv=records,
           reason='Synchronized driver manifest present; application child exited; all eight expected metric tables have complete block counts and stable hashes. Only CPU report postprocessing is terminated.')
    save(out/'postprocess-qualification.json',q)
    os.kill(a.pid,signal.SIGTERM)
    save(out/'candidate/collection.json',dict(status='completed',side='candidate',expected_kernel=kernel,
        groups='Roofline',task_alias=kernel,symbol_resolution=None,spec_sha256=sha(out/'spec.json'),
        discovery_sha256=sha(discovery),command=command,driver_sha256=sha(a.study/'method/hgkda/src/hgkda/runtime/paired_driver.py'),
        qualification='postprocess-qualification.json; full CSV recovered, vendor report process did not exit normally'))
    print(json.dumps(dict(status='qualified_complete_csv',kernel=kernel,blocks=blocks)))


if __name__=='__main__':main()
