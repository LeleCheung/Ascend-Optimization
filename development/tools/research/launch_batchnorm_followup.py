"""Start one model-authored follow-up after the fixed baseline completes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    base=Path('/data/hanle/ascend-optimization/runs')
    run=base/'midday-recovery-walltime-20261117/batch_norm_with_update_functional'
    audit=run/'formal-ca3ff529fa06/audit-final.json'
    while not audit.exists():time.sleep(10)
    baseline=json.loads(audit.read_bytes())['rows'][0]
    assert baseline['full_passed'], 'baseline failure must be diagnosed before optimization'
    shutil.copyfile(base/'night-research-20261116/no-simulator/model-settings.json',run/'model-settings.json')
    config=json.loads((run/'config.json').read_bytes())
    config.update(maximum_preflight_sources=8,maximum_formal_sources=8,
                  threshold=max(1.82,baseline['geo_mean']*1.02),
                  generation_target='Improve full walltime against the fixed current candidate while retaining all mutation, state and layout semantics.')
    (run/'config.json').write_text(json.dumps(config,indent=2))
    prompt=run/'optimization-prompt.md'
    prompt.write_text('''Load skill, task, current context, knowledge and full performance feedback.
The exact strict Triton delivery baseline just passed the full contract and supplements.
Historical native vs strict difference includes replacing implicit tensor copies with Triton.
Do not restore implicit PyTorch data movement or drop noncontiguous layout support.
Inspect actual current per-case timing and code and implement ONE generic improvement in
Triton work organization or redundant host metadata/launch work. Preserve running statistics,
training/evaluation behavior, dtype/precision, independent outputs, layouts and all side effects.
No hardcoded test-case values, no other backend computation, no public ABI changes.
Record a short decision before edit; current feedback is current-source evidence and older
results are background. Implement candidate.py and diagnosis.json only, then preflight and
return if awaiting_source_review. Do not claim causal performance without measurements.
Current baseline score: '''+str(baseline['geo_mean'])+'x CANN.\n')
    command=['/data/hanle/ascend-optimization/runtime/hgkda-release-venv/bin/python',
             '-m','hgkda.runtime.agent_runner','--config',str(run/'config.json'),
             '--model-settings',str(run/'model-settings.json'),'--invocation','performance-01',
             '--max-turns','24','--prompt-file',str(prompt)]
    with (run/'generation-01.log').open('xb') as log:
        result=subprocess.run(command,env=dict(os.environ,PYTHONPATH=str(Path(config['package'])/'src')),
                              stdout=log,stderr=subprocess.STDOUT)
    (run/'generation-exit.json').write_text(json.dumps(dict(returncode=result.returncode)))


if __name__=='__main__':main()
