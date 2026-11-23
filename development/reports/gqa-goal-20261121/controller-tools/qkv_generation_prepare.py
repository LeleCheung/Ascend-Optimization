"""Give a model qualified native evidence before its first QKV implementation edit."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

r=Path('/data/hanle/ascend-optimization/runs/qkv-simulator-20261118');w=r/'workspace'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
save=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
if '--wait' in sys.argv:
    while not (r/'native/collection-status.json').exists():time.sleep(10)
state=read(r/'native/collection-status.json')
sim_path=r/'native/simulator/summary.json'
if not sim_path.exists():sim_path=r/'native/simulator/instruction-summary.json'
sim=read(sim_path);assert sim['status']=='qualified_observation'
digest=sha(w/'candidate.py');assert sim['source_sha256']==digest
result=read(r/'native/result.json');target_id='transform_bias_rescale_qkv-timing-2'
target=next(x for x in result['per_workload'] if x['uuid']==target_id)
assert target['status']=='PASSED'
best=r/'best';best.mkdir();shutil.copyfile(w/'candidate.py',best/'candidate.py')
save(best/'selection.json',dict(source_sha256=digest,target_latency_ms=target['latency_ms'],
    target_speedup=target['speedup'],origin='raw native reference',candidate_admission='pending model self-contained implementation'))
save(r/'continuation.json',dict(status='native_evidence_ready',parent_source_sha256=digest,prior_display_rounds=0,
    attempts=[],qualified_rounds=[],best_source_sha256=digest,new_control_arm=False))
save(w/'previous-result.json',dict(source_sha256=digest,formal=result,scope='native upstream reference'))
shutil.copyfile(sim_path,w/'simulator-evidence.json')
comparison=read(r/'native/paired/comparison.json')
save(w/'current-profiling.json',dict(source_sha256=digest,comparison=comparison,scope='Available program rows; pipeline times overlap.'))
save(w/'knowledge.json',dict(mechanisms=[],scope='No task-specific prior implementation supplied; use measured evidence and skill methods.'))
artifacts=[];manifest=r/'native/paired/actual-ir/manifest.json';ir=w/'actual-ir';ir.mkdir()
if manifest.exists():
    for i,item in enumerate(read(manifest)['files']):
        path=manifest.parent/item['path']
        if path.suffix not in ['.ttir','.bcmlir']:continue
        assert sha(path)==item['sha256']
        dest=ir/(str(i)+'-'+path.name);shutil.copyfile(path,dest)
        artifacts.append(dict(path=str(dest.relative_to(w)),sha256=sha(dest)))
save(w/'context-brief.json',dict(source_sha256=digest,current_result='previous-result.json',
    current_profiling='current-profiling.json',current_simulator='simulator-evidence.json',actual_ir=artifacts))
save(w/'evidence-index.json',dict(source_sha256=digest,paths=['previous-result.json','current-profiling.json','simulator-evidence.json']+[x['path'] for x in artifacts]))
bridge=r/'decision_bridge.py';shutil.copyfile(r.parent/'upsample-performance-20261118/decision_bridge.py',bridge)
c=read(r/'config.json');c.update(agent_tool_path=str(bridge),required_simulator_evidence=dict(path='simulator-evidence.json',sha256=sha(w/'simulator-evidence.json')))
c['capture_entrypoint']='run';save(r/'config.json',c)
package=Path(c['package'])
shutil.copyfile(r/'method-manifest.json',r/'method-manifest-initial.json')
save(r/'method-manifest.json',dict(files=[dict(path=str(p.relative_to(package)),sha256=sha(p))
    for p in package.rglob('*') if p.is_file() and '__pycache__' not in p.parts]))
prompt='''Optimize transform_bias_rescale_qkv using the supplied original FlagGems source, full frozen contract and all analysis tools. No new control arm. Target timing-2: FP16 [8,512,3072], heads=16; preserve all other cases and supplementary checks. Read skills/hgkda/SKILL.md, task.json, contract.json, context-brief.json, current-profiling.json, simulator-evidence.json, relevant actual IR, and candidate.py. First identify what the simulator adds to profiling: locate the concrete hot operation and explain any observed waits, with partial-window limits. Pick one focused mechanism and predict both device changes and timing. Native baseline imports are allowed only for the reference; generated candidate must be self-contained strict Triton. Use torch only for allocation and metadata, maintain all input and bias strides, output semantics and numerical accuracy. Export run(qkv,qkv_bias,num_heads). Record interface/import compatibility separately from device optimization. No output cache, test-ID dispatch, external backend computation, or hidden host arithmetic.
Before editing candidate.py, fill diagnosis.json (source_sha256, evidence path/category/role/finding, hypothesis, alternatives, discriminating_observation, change, expected, validation, selected_layer, layer_reason, simulator_incremental_judgment). Register through the supplied decision bridge BEFORE any source edit. Read the current simulator successfully and cite actual observations. Use Edit only for candidate.py and diagnosis.json; Bash only the provided bridge. After producing one evaluable candidate, invoke preflight --source candidate.py; awaiting_source_review is the handoff to the host, so return with source unchanged for read-only review. No total optimization-round/tool/time budget. Each handoff preserves the candidate; host returns full actual results. Source/diagnosis prose should be professional Chinese. All implementation and compatibility changes are authored by you.
'''
(r/'prompt.md').write_text(prompt,encoding='utf8')
env=dict(os.environ,PYTHONPATH=str(Path(c['package'])/'src'))
cmd=[sys.executable,'-m','hgkda.runtime.agent_runner','--config',str(r/'config.json'),'--model-settings',str(r/'model-settings.json'),
    '--invocation','attempt-001','--prompt-file',str(r/'prompt.md')]
with (r/'generation-attempt-001.log').open('x') as log:
    proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
save(r/'process-attempt-001.json',dict(pid=proc.pid,command=cmd,starting_source_sha256=digest))
print(json.dumps(dict(pid=proc.pid,source_sha256=digest,target=target)))
