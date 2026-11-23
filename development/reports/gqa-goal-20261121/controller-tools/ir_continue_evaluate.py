"""Evaluate a reviewed continuation against its retained-IR comparator."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

from ir_campaign import evaluate, read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('--package', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    call = run / 'agent-calls' / args.invocation
    assert (call / 'model-usage.json').exists(), 'Wait for the model handoff before evaluation'
    review = read(call / 'ir-review.json')
    assert review['decision'] == 'approved'
    assert read(call / 'lowering-review.json')['status'] == 'lowering_review_passed'
    assert read(call / 'lowering-mechanism.json')['status'] == 'approved_for_formal'
    baseline = run / 'retained-baseline-verification.json'
    while not baseline.exists():
        time.sleep(10)
    verified = read(baseline)
    assert verified['status'] == 'verified' and verified['expected_binaries'] == verified['observed_binaries']
    original = run / 'evaluations/unchanged-ir/result.json'
    assert sha(original) == verified['result_sha256']
    evaluate(run, args.invocation)
    evaluation = run / 'evaluations' / args.invocation
    result = read(evaluation / 'result.json')
    checks = read(evaluation / 'supplementary-summary.json')
    complete = result['status'] == 'PASSED' and result['num_passed'] == result['num_workloads']
    qualified = complete and bool(checks) and all(r['status'] == 'passed' for r in checks.values())
    assessment = dict(status='host_retention_review_pending', complete_formal=complete,
         supplementary_passed=qualified, candidate=result, supplements=checks,
         retained_original=read(original), original_verification=verified,
         automatic_retention=False)
    save(run / ('assessment-' + args.invocation + '.json'), assessment)
    # Complete feedback is available to the next model call; input snapshots of
    # the finished call remain immutable.
    save(run / 'workspace/feedback.json', assessment)
    if qualified:
        scatter = read(run / 'config.json')['operator'] == 'scatter_add'
        collector = Path(__file__).with_name('ir_collect.py')
        assert collector.is_file()
        command = [sys.executable, str(collector),
            '--run', str(run), '--invocation', args.invocation, '--package', str(args.package)]
        if scatter:
            command += ['--kernel-name', '_scatter_add_sort_out_kernel', '--isolate-target']
        with (run / ('collection-' + args.invocation + '.log')).open('x') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        save(run / ('assessment-' + args.invocation + '.json'),
             dict(assessment, postcapture_returncode=process.returncode))


if __name__ == '__main__':
    main()
