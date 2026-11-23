"""Prepare equal, history-free generation workspaces for native-source comparison."""
import hashlib
import json
from pathlib import Path
import shutil

r=Path('/data/hanle/ascend-optimization/runs/native-upsample-contrast-20261118')
old=r.parent/'night-research-20261116'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

prompt='''Optimize upsample_bilinear2d from the supplied fixed original FlagGems source. The target is the full-call time for upsample_bilinear2d-timing-1 (FP16 [4,32,128,128] -> [4,32,256,256], align_corners=False); retain correctness on the full frozen contract and supplementary checks. Read task.json, contract.json, context-brief.json, knowledge.json, candidate.py and evidence listed in evidence-index.json. No historical best candidate or optimization solutions are provided. Do not read other directories. Each call proposes one focused mechanism change, not a bundle of speculative changes. Existing run entrypoint must be preserved; for the initial native source, adapt its exported function to run with the same signature. Native FlagGems imports may be replaced with self-contained equivalent Triton/metadata definitions as needed, but record compatibility edits separately from performance changes. No CANN/PyTorch/CPU computation or data copying in candidates; allocation, metadata and launch are allowed. No output caching or test-data fitting.
Before any edit to candidate.py, fill diagnosis.json with source_sha256, evidence, hypothesis, alternatives, discriminating_observation, change, expected and validation. Also include selected_layer (Triton, IR, pass, or compatibility), layer_reason, and simulator_incremental_judgment. Use exact evidence paths from the index; categories and roles follow the skill. Register via the decision tool bridge BEFORE editing. A failed decision must be corrected before editing. For the simulator arm, read simulator-evidence.json before deciding; explain what it adds, or explicitly say it adds no useful distinction. Never claim simulation evidence from source syntax alone. For the control arm, do not request or use simulation.
Use Read/Edit; Bash only for the supplied decision/preflight bridge. Make at most one new candidate this call, then request preflight. If awaiting_source_review, keep source unchanged and finish; the host will review and run checks and full performance. Return actual uncertainties. Do not stop the overall experiment at first correctness pass or first speedup: later calls supply real feedback. Preserve failed attempts. The shared maximum is 8 calls per arm, 48 tool turns per call. All guidance and changes are logged; do not edit contracts, evidence, knowledge, task or results.
'''
(r/'comparison-prompt.md').write_text(prompt)
for arm in ['no-simulator','with-simulator']:
    run=r/arm;run.mkdir();w=run/'workspace';w.mkdir()
    for name in ['candidate.py','starter.py']:(w/name).write_bytes((r/'upstream.py').read_bytes())
    shutil.copytree(r/'method/hgkda/skills/hgkda',w/'skills/hgkda')
    shutil.copytree(old/'no-simulator/reference-catalog',run/'reference-catalog')
    for name in ['operator.tar','request.json']:
        shutil.copyfile(old/'no-simulator'/name,run/name)
    request=json.loads((run/'request.json').read_bytes())
    request['implementation']['sources'][0]['content']=(r/'upstream.py').read_text()
    save(run/'request.json',request)
    shutil.copyfile(old/'no-simulator/model-settings.json',run/'model-settings.json')
    shutil.copyfile(old/'no-simulator/workspace/contract.json',w/'contract.json')
    save(w/'starter-provenance.json',dict(kind='fixed_native_flaggems',upstream_commit='d6a8eec473517a3d68157b208eb9c057eb1d4c50',source_sha256=sha(r/'upstream.py'),prior_best_source_provided=False))
    save(w/'knowledge.json',dict(mechanisms=[],policy='No task-specific historical optimization knowledge supplied to either arm; common skill constraints only.'))
    save(run/'controller-state.json',dict(status='awaiting_native_evidence'))
    package=r/'method/hgkda'
    save(run/'method-manifest.json',dict(files=[dict(path=str(p.relative_to(package)),sha256=sha(p)) for p in package.rglob('*') if p.is_file() and '__pycache__' not in p.parts]))
    c=json.loads((old/'no-simulator/config.json').read_bytes())
    c.update(run_directory=str(run),package=str(package),method_manifest=str(run/'method-manifest.json'),
             context_kind='research_contrast',goal_kind='performance_recovery_with_formal_feedback',
             kgs_server='http://127.0.0.1:19728',maximum_preflight_sources=8,maximum_formal_sources=8,
             agent_tool_path=str(r/'contrast_decision_bridge.py'),device_pythonpath='',
             experiment_goal=dict(kind='native_simulator_contrast',maximum_generation_calls=8,maximum_formal_sources=8,target_case='upsample_bilinear2d-timing-1'))
    c['job']['archive']=str(run/'operator.tar');c['job']['request']=str(run/'request.json')
    save(run/'config.json',c)
print('Prepared equal native-source workspaces; no generation started before evidence qualification.')
