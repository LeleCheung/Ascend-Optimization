"""Create an isolated performance-first continuation from a verified retained source."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    root = Path('/data/hanle/ascend-optimization')
    old = root/'runs/native-upsample-contrast-20261118'
    prior = old/'with-simulator'
    run = root/'runs/upsample-performance-20261118'
    run.mkdir(exist_ok=False)
    w = run/'workspace'; w.mkdir()
    package = run/'method/hgkda'
    shutil.copytree(root/'runtime/hgkda-performance-first-20261118/hgkda', package,
                    ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    def save(p, value):
        p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    source = prior/'agent-calls/round-06/output-source.py'
    digest = sha(source)
    assert digest == '6e7fed2de3d73b4174300829abb9f07e46b7f3372fb0904a6e79a1e52a7f32c3'
    for name in ['candidate.py','starter.py']:
        shutil.copyfile(source,w/name)
    best=run/'best';best.mkdir();shutil.copyfile(source,best/'candidate.py')
    for name in ['contract.json','knowledge.json']:
        shutil.copyfile(prior/'workspace'/name,w/name)
    shutil.copytree(package/'skills/hgkda',w/'skills/hgkda')
    for name in ['operator.tar','request.json','model-settings.json']:
        shutil.copyfile(prior/name,run/name)
    shutil.copyfile(old/'contrast_decision_bridge.py',run/'decision_bridge.py')
    shutil.copytree(prior/'source-reviews',run/'source-reviews')
    shutil.copytree(prior/('formal-'+digest[:12]),best/'original-validation')
    shutil.copytree(old/'final-replay/with-simulator',best/'final-replay')
    save(best/'selection.json',dict(source_sha256=digest,origin=str(source),display_round=2,
        target='upsample_bilinear2d-timing-1',policy='retain best fully qualified target performance; report all timing cases'))
    save(w/'starter-provenance.json',dict(kind='verified_retained_best',prior_best_source_provided=True,
        source_sha256=digest,origin=str(source),display_round=2))
    formal=json.loads((prior/('formal-'+digest[:12])/'evaluation/result.json').read_bytes())
    save(w/'previous-result.json',dict(source_sha256=digest,formal=formal))
    shutil.copyfile(prior/'simulator-6e7fed2d/summary.json',w/'simulator-evidence.json')
    comparison=json.loads((prior/'paired-6e7fed2d/comparison.json').read_bytes())
    save(w/'current-profiling.json',dict(source_sha256=digest,comparison=comparison,
        scope='Available program rows only; do not extrapolate their sums to the whole kernel.'))
    ir=w/'current-ir';ir.mkdir()
    verification=json.loads((prior/'attempts'/digest/'compiled/verification.json').read_bytes())
    artifacts=[]
    for i,entry in enumerate(verification['entries']):
        for item in entry['artifacts']:
            if item['kind'] not in ['ttir','bcmlir']:continue
            path=Path(item['path']);assert sha(path)==item['sha256']
            dest=ir/(str(i)+'-'+path.name);shutil.copyfile(path,dest)
            artifacts.append(dict(path=str(dest.relative_to(w)),sha256=sha(dest)))
    # Historical failures are explicitly separated from current-source measurements.
    history=w/'history';history.mkdir()
    for attempt in [7,8]:
        src=prior/f'agent-calls/round-{attempt:02d}/output-source.py'
        shutil.copyfile(src,history/f'attempt-{attempt}-source.py')
    save(history/'lessons.json',dict(scope='Historical exploration, separate source identities',observations=[
        dict(attempt=7,source_sha256=sha(history/'attempt-7-source.py'),result='numerical failure: 28/30 supplementary; packed half writeback produced incorrect values/NaNs',
             source='attempt-7-source.py',next_question='Determine whether bit representation, alignment, or compiler lowering caused the failure before reusing packing.'),
        dict(attempt=8,source_sha256=sha(history/'attempt-8-source.py'),result='full correctness passed; target 4.323741ms; scalar activity 95.7%; 16x128 scalar load loop in bcmlir',
             source='attempt-8-source.py',next_question='Preserve affine contiguous addresses; investigate boundary handling separately.')]))
    save(w/'context-brief.json',dict(source_sha256=digest,current_result='previous-result.json',
        current_profiling='current-profiling.json',current_simulator='simulator-evidence.json',actual_ir=artifacts,
        historical_lessons='history/lessons.json',scope='Continue verified best with all tools; no new comparison arm.'))
    save(w/'evidence-index.json',dict(source_sha256=digest,paths=['previous-result.json','current-profiling.json',
        'simulator-evidence.json','history/lessons.json']+[a['path'] for a in artifacts]))
    c=json.loads((prior/'config.json').read_bytes())
    c.update(run_directory=str(run),package=str(package),method_manifest=str(run/'method-manifest.json'),
        context_kind='performance_continuation',optimization_policy='performance_first',
        maximum_preflight_sources=0,maximum_formal_sources=0,agent_tool_path=str(run/'decision_bridge.py'),
        experiment_goal=dict(kind='performance_continuation',target_case='upsample_bilinear2d-timing-1',new_control_arm=False),
        required_simulator_evidence=dict(path='simulator-evidence.json',sha256=sha(w/'simulator-evidence.json')))
    c['job']['archive']=str(run/'operator.tar');c['job']['request']=str(run/'request.json')
    save(run/'config.json',c)
    save(run/'method-manifest.json',dict(files=[dict(path=str(p.relative_to(package)),sha256=sha(p))
        for p in package.rglob('*') if p.is_file() and '__pycache__' not in p.parts]))
    prompt='''Continue upsample_bilinear2d from the supplied verified best. Use all available tools; no control arm. Optimize target timing-1 while preserving the frozen full contract and all supplementary checks. Read skills/hgkda/SKILL.md, task.json, contract.json, context-brief.json, current-profiling.json, simulator-evidence.json and history/lessons.json before deciding. Read relevant IR snippets and candidate.py to explain mechanisms. Simulator data is a partial capture: distinguish visible observations from whole-kernel timing. Historical failed-source observations have separate hashes.
Choose one focused performance hypothesis from the evidence. Investigate data rearrangement/writeback overhead or another concretely observed bottleneck. Historical packing had numerical failure; historical boundary clamping scalarized loads. Use these observations to avoid repeating mistakes. Preserve working generic paths and ABI. You author all candidate edits; Codex only supplies evidence and reviews. Strict Triton computation; torch only allocations/metadata. No test fitting, output caching, or alternate backend computation.
Before editing candidate.py, use Edit to fill diagnosis.json with source_sha256, evidence entries (path/category/role/finding), hypothesis, alternatives, discriminating_observation, change, expected, validation, selected_layer, layer_reason, simulator_incremental_judgment. Register using decision --record diagnosis.json BEFORE edits. Read the current simulator report successfully and cite its actual contribution. Edit only candidate.py and diagnosis.json. Bash only the supplied bridge. Keep one evaluable hypothesis per handoff; no workflow limit on total attempts or optimization time. After edits call preflight --source candidate.py. On awaiting_source_review return unchanged for host review and full evaluation. This handoff preserves all work and the host continues from actual feedback. Source comments and diagnosis use professional Chinese.
'''
    (run/'prompt.md').write_text(prompt,encoding='utf8')
    save(run/'continuation.json',dict(parent_study=str(old),parent_source_sha256=digest,
        prior_display_rounds=3,status='prepared',new_control_arm=False,default_start='best/candidate.py',
        attempts=[],qualified_rounds=[],best_source_sha256=digest))
    save(run/'controller-state.json',dict(status='prepared'))
    print(json.dumps(dict(run=str(run),source_sha256=digest)))


if __name__=='__main__':main()
