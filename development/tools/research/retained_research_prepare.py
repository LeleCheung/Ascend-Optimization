"""Prepare a source-preserving continuation with explicit operator and target case."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--operator-directory', type=Path, required=True)
p.add_argument('--package', type=Path, required=True)
p.add_argument('--environment', type=Path, required=True)
p.add_argument('--run', type=Path, required=True)
p.add_argument('--template-run', type=Path, required=True)
p.add_argument('--target-case', required=True)
a = p.parse_args()
subprocess.run([sys.executable, '-m', 'hgkda.runtime.prepare',
    '--operator-directory', str(a.operator_directory), '--package', str(a.package),
    '--environment', str(a.environment), '--run-directory', str(a.run)], check=True)
r = a.run
read = lambda p: json.loads(p.read_bytes())
def save(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
digest = hashlib.sha256((r/'workspace/candidate.py').read_bytes()).hexdigest()
c = read(r/'config.json')
c.update(capture_entrypoint='run', capture_warmup=0,
    context_kind='performance_continuation', optimization_policy='performance_first',
    goal_kind='performance_recovery_with_formal_feedback',
    agent_tool_path=str(r/'decision_bridge.py'),
    experiment_goal=dict(kind='simulator_guided_optimization', target_case=a.target_case, new_control_arm=False))
timings = [json.loads(s) for s in (r/'reference-catalog/timing.jsonl').read_text().splitlines() if s.strip()]
assert any(x['name'] == a.target_case for x in timings)
save(r/'config.json', c)
for name in ['decision_bridge.py', 'model-settings.json', 'regions_optimized.py', 'contrast_simulator_summary.py']:
    shutil.copyfile(a.template_run/name, r/name)
(r/'method').mkdir()
(r/'method/hgkda').symlink_to(Path(c['package']), target_is_directory=True)
save(r/'continuation.json', dict(status='starting_validation', parent_source_sha256=digest,
    prior_display_rounds=0, attempts=[], qualified_rounds=[], best_source_sha256=digest,
    new_control_arm=False, origin='retained delivery continuation'))
save(r/'study.json', dict(operator=c['operator'], source_sha256=digest,
    target_case=a.target_case, origin=str(a.operator_directory),
    service_ownership='borrowed campaign service; do not terminate from this run'))
cmd = [sys.executable, '-m', 'hgkda.runtime.evaluate', '--config', str(r/'config.json'),
       '--source', str(r/'workspace/candidate.py'), '--review', str(r/'source-reviews'/(digest+'.json'))]
with (r/'starting-validation.log').open('x') as log:
    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True)
save(r/'starting-validation-process.json', dict(pid=process.pid, command=cmd))
print(json.dumps(dict(run=str(r), source_sha256=digest, pid=process.pid)))
