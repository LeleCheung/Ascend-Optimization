"""Continue a failed candidate inside the same hypothesis, using actual failure evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--invocation',required=True)
p.add_argument('--failure-file',type=Path,help='Source-bound failed supplementary result from formal validation')
a=p.parse_args();r=a.run;w=r/'workspace';read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
save=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
digest=sha(w/'candidate.py');failed=read(a.failure_file or w/'latest-preflight.json')
if a.failure_file:
    assert failed['passed']<failed['total']
    failed=dict(failed,status='failed',failure_stage='formal supplementary',original_failure=str(a.failure_file),failure_sha256=sha(a.failure_file))
assert failed['source_sha256']==digest and failed['status']=='failed'
diagnostic_logs=[]
for path in (r/'attempts'/digest).glob('*.log'):
    diagnostic_logs.append(dict(path=str(path),sha256=sha(path),
        scope='Full log from this frozen source attempt; initial compiler diagnostics may precede the traceback tail.'))
failed['full_diagnostic_logs']=diagnostic_logs
c=read(r/'config.json');save(r/('config-before-'+a.invocation+'.json'),c)
c.pop('required_simulator_evidence',None)
save(r/'config.json',c)
save(w/'previous-result.json',failed)
artifacts=[];verification=r/'attempts'/digest/'compiled/verification.json'
if verification.exists():
    v=read(verification);assert v['source_sha256']==digest
    folder=w/'current-ir'/digest[:12];folder.mkdir(parents=True,exist_ok=True)
    for i,entry in enumerate(v['entries']):
        for item in entry['artifacts']:
            if item['kind'] not in ['ttir','bcmlir']:continue
            path=Path(item['path']);assert sha(path)==item['sha256']
            dest=folder/(str(i)+'-'+path.name);shutil.copyfile(path,dest)
            artifacts.append(dict(path=str(dest.relative_to(w)),sha256=sha(dest)))
else:
    attempt=r/'attempts'/digest
    frozen=attempt/'candidate.py'
    assert sha(frozen)==digest
    folder=w/'failed-ir'/digest[:12];folder.mkdir(parents=True,exist_ok=True)
    for kind in ['ttir','bcmlir']:
        for index,path in enumerate((attempt/'triton-cache').rglob('*.'+kind)):
            if str(frozen) not in path.read_text(errors='replace'):continue
            dest=folder/(path.parent.name[:12]+'-'+str(index)+'-'+path.name)
            shutil.copyfile(path,dest)
            artifacts.append(dict(path=str(dest.relative_to(w)),sha256=sha(dest),scope='Actual intermediate compiler artifact from failed compilation; no device binary or performance claim.'))
save(w/'context-brief.json',dict(source_sha256=digest,current_result='previous-result.json',actual_ir=artifacts,
    current_profiling=None,current_simulator=None,
    historical_evidence='current-profiling.json and simulator-evidence.json belong to the original source; retain as historical context only.',
    phase='repair within prior performance hypothesis; preserve changed tiling and indexing'))
save(r/('branch-'+a.invocation+'.json'),dict(starting_source_sha256=digest,reason='Repair the failed exploration using current-source feedback; preserve the qualified best.',
    kept_best=read(r/'best/selection.json')['source_sha256']))
prompt='''Repair the current failed candidate within its existing performance hypothesis. Read task.json, contract.json, context-brief.json, previous-result.json and candidate.py. Read the beginning of full_diagnostic_logs for the first compiler diagnostic as well as the traceback tail. Identify the actual failure stage and exact diagnostic. For compiler resource failures, inspect the reported memory requirement and adjust tile geometry or buffer lifetime with a focused change. For numerical failures, inspect arithmetic order, precision, masks and independent-output semantics. Preserve verified numerical fixes and generic layout paths. Use the current feedback as the authority for which checks passed.
Current-source simulator and profiling evidence are unavailable for this repair. Their existing files describe earlier sources and remain historical context. Record the actual diagnostic, proposed repair, expected effect and validation in diagnosis.json; register through the decision bridge before editing candidate.py. All candidate changes are authored by you. Use Edit for candidate.py and diagnosis.json; device checks use the supplied bridge. Invoke preflight and hand off unchanged on awaiting_source_review. Strict Triton computation and allocation/metadata-only host operations remain required. Keep the candidate self-contained with permitted imports. No total search or tool-call budget.
'''
prompt_path=r/(a.invocation+'-prompt.md');prompt_path.write_text(prompt,encoding='utf8')
cmd=[sys.executable,'-m','hgkda.runtime.agent_runner','--config',str(r/'config.json'),'--model-settings',str(r/'model-settings.json'),
    '--invocation',a.invocation,'--prompt-file',str(prompt_path)]
with (r/('generation-'+a.invocation+'.log')).open('x') as log:
    proc=subprocess.Popen(cmd,env=dict(os.environ,PYTHONPATH=str(Path(c['package'])/'src')),stdout=log,stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,start_new_session=True)
save(r/('process-'+a.invocation+'.json'),dict(pid=proc.pid,command=cmd,source_sha256=digest))
print(json.dumps(dict(pid=proc.pid,source_sha256=digest)))
