"""Verify the retained-IR comparator before asking the model for a new edit."""
import argparse
from pathlib import Path
import subprocess
import sys

from ir_campaign import evaluate, read, save, sha
from ir_stability import binaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', default='ir-001')
    parser.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    config = read(run / 'config.json')
    baseline = run / 'evaluations/unchanged-ir'
    if not (baseline / 'result.json').exists():
        evaluate(run, 'unchanged-ir', resume=(baseline / 'request.json').exists())
    result = read(baseline / 'result.json')
    assert result['status'] == 'PASSED' and result['num_passed'] == result['num_workloads']
    parent = Path(config['continuation_parent']) / 'evaluations' / config['continuation_invocation']
    expected = binaries(parent / 'compiler-receipts')
    observed = binaries(baseline / 'compiler-receipts')
    assert expected == observed, 'Continuation baseline changed the retained binary'
    target_case = config['experiment_goal']['target_case']
    target = next(r for r in result['per_workload'] if r['uuid'] == target_case and r['phase'] == 'timing')
    save(run / 'retained-baseline-verification.json', dict(status='verified',
         parent=str(parent), expected_binaries=expected, observed_binaries=observed,
         complete_passed=result['num_passed'], target=target,
         result_sha256=sha(baseline / 'result.json'),
         scope='Complete contract replay and byte-identical retained binaries; measurement variation separately preserved'))
    if args.baseline_only:
        return
    assert all((run / 'workspace' / name).is_file() for name in
               ['profiling.json', 'hotspots.json', 'timeline.json']), 'Collect bound evidence before model authoring'
    save(run / 'workspace/baseline-result.json', dict(current=result, target=target))
    invocation = args.invocation
    assert not (run / 'agent-calls' / invocation).exists()
    with (run / ('generation-' + invocation + '.log')).open('x') as log:
        process = subprocess.run([sys.executable, str(Path(__file__).with_name('ir_model.py')),
            '--run', str(run), '--invocation', invocation, 'generate'],
            stdout=log, stderr=subprocess.STDOUT)
    save(run / 'driver-handoff.json', dict(status='host_review_required',
         returncode=process.returncode, invocation=invocation,
         model_status=read(run / 'status.json')))


if __name__ == '__main__':
    main()
