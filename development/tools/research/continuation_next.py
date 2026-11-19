"""Start a performance hypothesis from retained best with both hotspot and timeline evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--invocation',required=True)
a=p.parse_args();r=a.run;w=r/'workspace';read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
save=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
best=read(r/'best/selection.json');digest=best['source_sha256'];assert sha(r/'best/candidate.py')==digest
if sha(w/'candidate.py')!=digest:
    save(r/('restore-'+a.invocation+'.json'),dict(from_sha256=sha(w/'candidate.py'),to_sha256=digest,reason='New hypothesis starts from retained qualified best.'))
    shutil.copyfile(r/'best/candidate.py',w/'candidate.py')
pair=r/('paired-'+digest[:8]);sim=r/('simulator-'+digest[:8]);spec=read(pair/'spec.json');assert spec['source_sha256']==digest
summary_path=sim/'summary.json'
if not summary_path.exists():summary_path=sim/'instruction-summary.json'
summary=read(summary_path);assert summary['source_sha256']==digest
relative=read(sim/'relative-timeline-summary.json');assert relative['source_sha256']==digest
by_core={x['core']:x for x in relative['reports']}
for report in summary['reports']:
    report['absolute_timeline']=report.get('timeline')
    report['timeline']=by_core[report['core']]['timeline']
summary['relative_analysis']=dict(analyzer_sha256=relative['analyzer_sha256'],regions_sha256=relative['regions_sha256'],
    scope=relative['scope'],absolute_time_qualified=False)
save(w/'simulator-evidence.json',summary)
save(w/'current-profiling.json',dict(source_sha256=digest,comparison=read(pair/'comparison.json'),scope='Available program rows; overlapping pipelines.'))
history=[]
for item in read(r/'continuation.json').get('attempts',[]):
    assessment=Path(item['assessment'])/'result.json'
    value=read(assessment)
    entry=dict(invocation=item['invocation'],source_sha256=item['source_sha256'],qualified=item['qualified'],
        retained=item.get('retained'),target_latency_ms=item.get('target_latency_ms'))
    pre=value.get('preflight',{})
    if pre.get('status')=='failed':
        entry['failure']=dict(stage='preflight',failed_cases=pre.get('failed_cases'),log_tail=pre.get('log_tail'))
    formal_dir=r/('formal-'+item['source_sha256'][:12])
    failures=[]
    for path in formal_dir.glob('*.json'):
        document=read(path)
        if isinstance(document,dict) and isinstance(document.get('passed'),int) and document.get('total',0)>document['passed']:
            failures.append(dict(check=path.name,source_sha256=document.get('source_sha256'),passed=document['passed'],total=document['total'],
                cases=[x for x in document.get('cases',[]) if not x.get('passed',False)]))
    if failures:entry['formal_check_failures']=failures
    # Keep postchange observations attached to their historical source. A new
    # hypothesis can compare a regression with the retained source explicitly.
    historical_digest=item['source_sha256']
    historical_paths={
        'profiling':r/('paired-'+historical_digest[:8])/'comparison.json',
        'hotspots':r/('simulator-'+historical_digest[:8])/'summary.json',
        'timeline':r/('simulator-'+historical_digest[:8])/'relative-timeline-summary.json',
    }
    entry['postchange_evidence']={kind:dict(path=str(path),sha256=sha(path))
        for kind,path in historical_paths.items() if path.exists()}
    history.append(entry)
save(w/'exploration-feedback.json',dict(scope='Historical candidates with distinct source identities; current measurements remain separately bound.',attempts=history))
formal=r/('formal-'+digest[:12]);save(w/'previous-result.json',dict(source_sha256=digest,formal=read(formal/'evaluation/result.json')))
verification=r/'attempts'/digest/'compiled/verification.json';artifacts=[]
folder=w/'current-ir'/digest[:12];folder.mkdir(parents=True,exist_ok=True)
for i,entry in enumerate(read(verification)['entries']):
    for item in entry['artifacts']:
        if item['kind'] not in ['ttir','bcmlir']:continue
        path=Path(item['path']);assert sha(path)==item['sha256']
        dest=folder/(str(i)+'-'+path.name);shutil.copyfile(path,dest);artifacts.append(dict(path=str(dest.relative_to(w)),sha256=sha(dest)))
target_manifest=pair/'actual-ir/manifest.json'
target_ir=[]
if target_manifest.exists():
    manifest=read(target_manifest);assert manifest['source_sha256']==digest
    for i,item in enumerate(manifest['files']):
        path=target_manifest.parent/item['path']
        if path.suffix not in ['.ttir','.bcmlir']:continue
        assert sha(path)==item['sha256']
        dest=folder/('target-'+str(i)+'-'+path.name);shutil.copyfile(path,dest)
        target_ir.append(dict(path=str(dest.relative_to(w)),sha256=sha(dest),scope='Actual profiling target-case compiler artifact'))
save(w/'context-brief.json',dict(source_sha256=digest,current_result='previous-result.json',current_profiling='current-profiling.json',
    current_simulator='simulator-evidence.json',target_ir=target_ir,actual_ir=artifacts,historical_feedback='exploration-feedback.json',phase='new performance hypothesis from retained best'))
c=read(r/'config.json');c['required_simulator_evidence']=dict(path='simulator-evidence.json',sha256=sha(w/'simulator-evidence.json'));save(r/'config.json',c)
prompt='''Continue optimizing the retained fully qualified best candidate. Read task.json, contract.json, context-brief.json, previous-result.json, current-profiling.json, simulator-evidence.json, candidate.py and relevant actual IR. Both hotspot and pipeline analyses are required inputs. The simulator reports include normalized per-core timelines: productive_ratio, wait_only_ratio, idle_ratio. These are fractions of a partial observed common-clock window, with absolute_time_qualified=false; use them to distinguish hypotheses, not as full-device time or proof of dependency stalls. State what the hotspots add and what the waiting/observable idle adds; if a method adds no useful distinction, say so explicitly while retaining its observations.
Select one focused mechanism with a predicted measurable effect. Preserve all established numerical fixes, layout and special-case semantics. No test fitting, output caching or alternate-backend computation. Host allocation/metadata and Triton compute only. Before editing candidate.py, fill diagnosis.json with the current source SHA, evidence path/category/role/finding, hypothesis, alternatives, discriminating_observation, change, expected, validation, selected_layer, layer_reason and simulator_incremental_judgment. Register through decision --record diagnosis.json before edits. Use Edit only for candidate.py and diagnosis.json; Bash only the supplied bridge. After one evaluable hypothesis invoke preflight --source candidate.py; on awaiting_source_review hand off unchanged for host review and complete formal evaluation. No total search/tool/time budget; this handoff is a validation boundary. All candidate changes are authored by you.
'''
prompt+='\nRead exploration-feedback.json before choosing a new hypothesis. Preserve source identities: past failures are constraints and lessons, while current profiling and simulator observations belong only to the retained source. Address prior numerical failures generically rather than repeating the same rejected transformation.\n'
prompt+='Historical postchange_evidence entries link observations of qualified explorations, including regressions. Use relevant comparisons to distinguish mechanisms; cite the source SHA of each observation and keep postchange interpretation separate from evidence read before that historical edit.\n'
prompt_path=r/(a.invocation+'-prompt.md');prompt_path.write_text(prompt,encoding='utf8')
cmd=[sys.executable,'-m','hgkda.runtime.agent_runner','--config',str(r/'config.json'),'--model-settings',str(r/'model-settings.json'),
    '--invocation',a.invocation,'--prompt-file',str(prompt_path)]
with (r/('generation-'+a.invocation+'.log')).open('x') as log:
    proc=subprocess.Popen(cmd,env=dict(os.environ,PYTHONPATH=str(Path(c['package'])/'src')),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
save(r/('process-'+a.invocation+'.json'),dict(pid=proc.pid,command=cmd,starting_source_sha256=digest))
print(json.dumps(dict(pid=proc.pid,source_sha256=digest)))
