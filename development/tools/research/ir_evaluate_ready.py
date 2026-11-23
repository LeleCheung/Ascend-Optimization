"""评测已完成宿主审查及CPU编译的IR，完整通过后采集双分析。"""
import argparse
from pathlib import Path
import subprocess
import sys

from ir_campaign import evaluate, read, save, sha
from ir_initialize import verify_noop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--kernel-name')
    parser.add_argument('--isolate-target', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    call = run / 'agent-calls' / args.invocation
    review = read(call / 'ir-review.json')
    assert review['decision'] == 'approved'
    assert read(call / 'lowering-review.json')['status'] == 'lowering_review_passed'
    original = verify_noop(run)
    evaluate(run, args.invocation)
    output = run / 'evaluations' / args.invocation
    result = read(output / 'result.json')
    checks = read(output / 'supplementary-summary.json')
    complete = result['status'] == 'PASSED' and result['num_passed'] == result['num_workloads']
    supplements = bool(checks) and all(row['status'] == 'passed' for row in checks.values())
    config = read(run / 'config.json')
    target = next((row for row in result.get('per_workload', [])
        if row['uuid'] == config['experiment_goal']['target_case'] and row['phase'] == 'timing'), None)
    record = dict(status='host_assessment_pending', invocation=args.invocation, complete_formal=complete,
        supplementary_passed=supplements, candidate_target=target,
        baseline_target=original['baseline_target'], unchanged_ir_target=original['unchanged_ir_target'],
        review_sha256=sha(call / 'ir-review.json'), automatic_retention=False)
    save(run / ('assessment-' + args.invocation + '.json'), record)
    if complete and supplements:
        command = [sys.executable, str(Path(__file__).with_name('ir_collect.py')), '--run', str(run),
            '--invocation', args.invocation, '--package', str(args.package)]
        if args.kernel_name:
            command += ['--kernel-name', args.kernel_name]
        if args.isolate_target:
            assert args.kernel_name, 'Isolated collection requires an explicit target'
            command += ['--isolate-target']
        with (run / ('collection-' + args.invocation + '.log')).open('x') as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        save(run / ('assessment-' + args.invocation + '.json'),
            dict(record, postcapture_returncode=process.returncode))


if __name__ == '__main__':
    main()
