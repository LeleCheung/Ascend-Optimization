"""Queue ABBA only when the active policy calls for measurement diagnostics."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')


def screen_improvement(run, invocation, summary):
    evaluation = run / 'evaluations' / invocation
    candidate = read(evaluation / 'result.json')
    qualified = (candidate['status'] == 'PASSED' and candidate['num_passed'] == candidate['num_workloads']
                 and summary and all(row['status'] == 'passed' for row in summary.values()))
    if not qualified:
        return dict(status='not_run_candidate_unqualified', invocation=invocation)
    config = read(run / 'config.json')
    name = config['experiment_goal']['target_case']
    cases = lambda result: next(row for row in result['per_workload']
                               if row['uuid'] == name and row['phase'] == 'timing')
    target = cases(candidate)
    original = cases(read(run / 'evaluations/unchanged-ir/result.json'))
    historical = cases(read(Path(config['continuation_parent']) / 'evaluations' /
                            config['continuation_invocation'] / 'result.json'))
    improved = target['latency_ms'] < min(original['latency_ms'], historical['latency_ms'])
    return dict(status='qualified_target_improvement' if improved else 'not_run_no_target_improvement',
        invocation=invocation, candidate=target, current_stage_retained=original,
        historical_retained=historical, automatic_retention=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('--kgs-root', type=Path, required=True)
    parser.add_argument('--require-improvement', action='store_true')
    parser.add_argument('--reason', help='Concrete measurement concern requiring ABBA')
    args = parser.parse_args()
    policy_path = args.run / 'validation-policy.json'
    if policy_path.exists():
        policy = read(policy_path)
        if policy.get('abba') == 'conditional' and not args.reason:
            save(args.run / ('stability-screen-' + args.invocation + '.json'),
                dict(status='not_run_no_measurement_concern', invocation=args.invocation,
                     policy='conditional', reason='Routine acceptance uses the fixed contract'))
            return
    checks = args.run / 'evaluations' / args.invocation / 'supplementary-summary.json'
    while not checks.exists():
        time.sleep(10)
    summary = read(checks)
    if args.require_improvement:
        screened = screen_improvement(args.run, args.invocation, summary)
        save(args.run / ('stability-screen-' + args.invocation + '.json'), screened)
        if screened['status'] != 'qualified_target_improvement':
            return
    assert summary and all(row['status'] == 'passed' for row in summary.values())
    subprocess.run([sys.executable, str(Path(__file__).with_name('ir_stability.py')),
        '--run', str(args.run), '--invocation', args.invocation, '--kgs-root', str(args.kgs_root)], check=True)


if __name__ == '__main__':
    main()
