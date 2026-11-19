"""Start one shared compatibility-only model call, separate from both study arms."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

r=Path('/data/hanle/ascend-optimization/runs/native-upsample-contrast-20261118')
run=r/'compatibility';run.mkdir()
w=run/'workspace';w.mkdir()
for name in ['candidate.py','starter.py','contract.json','knowledge.json','starter-provenance.json']:
    shutil.copyfile(r/'no-simulator/workspace'/name,w/name)
shutil.copytree(r/'no-simulator/workspace/skills',w/'skills')
for name in ['operator.tar','request.json','model-settings.json','method-manifest.json']:
    shutil.copyfile(r/'no-simulator'/name,run/name)
c=json.loads((r/'no-simulator/config.json').read_bytes());c['run_directory']=str(run)
c['method_manifest']=str(run/'method-manifest.json');c['job']['archive']=str(run/'operator.tar');c['job']['request']=str(run/'request.json')
c['experiment_goal']={'kind':'shared_compatibility_only','maximum_generation_calls':2,'performance_optimization':False}
c['generation_target']='Make the original function compile and satisfy the unchanged contract, without performance optimization; preserve source ancestry.'
(run/'config.json').write_text(json.dumps(c,indent=2))
(run/'controller-state.json').write_text(json.dumps({'status':'prepared'}))
raw=(r/'kgs.log').read_text(errors='replace');errors=[line for line in raw.splitlines() if 'error:' in line or 'Pipeline failed' in line or 'tl.store' in line]
digest=hashlib.sha256((w/'candidate.py').read_bytes()).hexdigest()
(w/'compiler-evidence.json').write_text(json.dumps({'source_sha256':digest,'status':'native_evaluation_still_running','errors':list(dict.fromkeys(errors))[-12:],'scope':'Compilation errors observed in the untouched native baseline. Full 24-case result pending.'},indent=2))
(w/'context-brief.json').write_text(json.dumps({'source_sha256':digest,'phase':'shared compatibility','simulator_available':False,'full_baseline_evaluation':'pending; no performance claims'},indent=2))
(w/'evidence-index.json').write_text(json.dumps({'files':['candidate.py','compiler-evidence.json','contract.json']},indent=2))
prompt='''This is a SHARED COMPATIBILITY phase before a simulator/control experiment, not an optimization arm. Read skills/hgkda/SKILL.md, task.json, contract.json, candidate.py, compiler-evidence.json. The native fixed FlagGems module is untouched. Make the smallest self-contained STRICT TRITON adaptation needed to compile in Triton-Ascend and expose run(input,output_size,align_corners=False,scales_h=None,scales_w=None). Preserve algorithm, loops, interpolation formula and intended launch geometry wherever possible. Do not optimize performance or inject historical implementations. Replace FlagGems-only metadata/dependencies with equivalent local Triton/host metadata if required by the allowed imports. Preserve all supported semantics. You may read only this workspace. No historical best is supplied.
Before editing candidate.py, fill diagnosis.json including source_sha256, evidence entries (paths relative to RUN, e.g. workspace/compiler-evidence.json), hypothesis, change, expected, validation, alternatives, selected_layer='compatibility', layer_reason, simulator_incremental_judgment='No simulation in common compatibility phase'. Register with the provided decision bridge before editing. Then implement ONE candidate, request preflight, and stop if awaiting_source_review. Use Read/Edit; Bash only for the provided bridge. No target computation in PyTorch/CPU/CANN, no native DMA. Describe exactly what changed for compatibility and what remains uncertain. No performance claims. Maximum 48 tool turns.'''
(run/'prompt.md').write_text(prompt)
cmd=[sys.executable,'-m','hgkda.runtime.agent_runner','--config',str(run/'config.json'),'--model-settings',str(run/'model-settings.json'),'--invocation','compatibility-01','--max-turns','48','--prompt-file',str(run/'prompt.md')]
with (run/'generation.log').open('x') as log:
    process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH=str(r/'method/hgkda/src')),start_new_session=True)
(run/'process.json').write_text(json.dumps({'pid':process.pid,'command':cmd,'phase':'common compatibility; not counted as simulator benefit'},indent=2))
print(process.pid)
