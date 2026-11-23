"""保留首次接口故障，重测原IR并继续已审查的IR候选。"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

from ir_campaign import evaluate, read, save, sha
from ir_initialize import verify_noop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('--package', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    save(run / 'noop-recovery-status.json', dict(status='waiting_for_initial_original_ir_result'))
    while not (run / 'initialization-error.json').exists() and not (run / 'original-ir-verification.json').exists():
        time.sleep(5)
    if not (run / 'original-ir-verification.json').exists():
        archive = run / 'evaluation-history/unchanged-ir-text-roundtrip'
        if not archive.exists():
            result = read(run / 'evaluations/unchanged-ir/result.json')
            assert result['status'] != 'PASSED', 'Inspect a verification-only failure before retrying'
            failures = [row for row in result.get('per_workload', []) if row.get('status') != 'PASSED']
            assert failures and all('Override did not reach the cached compiler stage' in row.get('log', '')
                for row in failures), 'Recovery requires the identified textual-round-trip assertion'
            archive.parent.mkdir(exist_ok=True)
            (run / 'evaluations/unchanged-ir').rename(archive)
            save(archive / 'relocation.json', dict(previous_path=str(run / 'evaluations/unchanged-ir'),
                stored_path=str(archive), timestamp=datetime.now(timezone.utc).isoformat(),
                reason='Native parser SSA renaming requires separate raw and parsed-stage identity'))
        save(run / 'noop-recovery-status.json', dict(status='replaying_original_ir', archive=str(archive)))
        evaluate(run, 'unchanged-ir', resume=(run / 'evaluations/unchanged-ir/request.json').exists())
        verify_noop(run)
    review_path = run / 'agent-calls' / args.invocation / 'ir-review.json'
    review = read(review_path)
    assert review['decision'] == 'approved'
    save(run / 'noop-recovery-status.json', dict(status='evaluating_reviewed_ir', invocation=args.invocation,
        review_sha256=sha(review_path)))
    evaluate(run, args.invocation)
    result = read(run / 'evaluations' / args.invocation / 'result.json')
    checks = read(run / 'evaluations' / args.invocation / 'supplementary-summary.json')
    config = read(run / 'config.json')
    target = config['experiment_goal']['target_case']
    candidate = next((row for row in result.get('per_workload', [])
        if row['uuid'] == target and row['phase'] == 'timing'), None)
    original = read(run / 'original-ir-verification.json')
    complete = result['status'] == 'PASSED' and result['num_passed'] == result['num_workloads']
    checks_passed = bool(checks) and all(row['status'] == 'passed' for row in checks.values())
    direction = complete and checks_passed and candidate['latency_ms'] < min(
        original['baseline_target']['latency_ms'], original['unchanged_ir_target']['latency_ms'])
    record = dict(status='candidate_requires_assessment', invocation=args.invocation,
        complete_formal=complete, supplementary_passed=checks_passed,
        baseline_target=original['baseline_target'], unchanged_ir_target=original['unchanged_ir_target'],
        candidate_target=candidate, provisional_direction=bool(direction),
        scope='Host assessment, small-gain stability and post-capture verification still apply')
    save(run / 'noop-recovery-status.json', record)
    if direction:
        command = [sys.executable, str(Path(__file__).with_name('ir_collect.py')), '--run', str(run),
            '--invocation', args.invocation, '--package', str(args.package)]
        with (run / ('collection-' + args.invocation + '.log')).open('x') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        save(run / 'noop-recovery-status.json', dict(record, postcapture_returncode=process.returncode))


if __name__ == '__main__':
    main()
