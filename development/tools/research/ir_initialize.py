"""Capture frozen bests and verify unchanged-IR execution before model editing."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from ir_campaign import capture, evaluate, read, save, seed, sha


def qualified(result):
    return result.get('status') == 'PASSED' and result.get('num_passed') == result.get('num_workloads')


def verify_noop(run):
    baseline_path = run / 'evaluations/frozen-baseline/result.json'
    baseline_scope = 'Current full frozen-source baseline'
    if not baseline_path.exists():
        parent = read(run / 'retained-parent.json')
        assert sha(Path(parent['original_source'])) == sha(run / 'workspace/frozen.py') == parent['source_sha256']
        baseline_path = Path(parent['evaluation'])
        baseline_scope = 'Previously accepted identical source and fixed contract; historical timing retained'
    baseline = read(baseline_path)
    noop = read(run / 'evaluations/unchanged-ir/result.json')
    assert qualified(baseline) and qualified(noop), 'Original IR requires complete formal correctness'
    inventory = read(run / 'workspace/ir-inventory.json')['entries']
    expected = {row['ast_sha256'] for row in inventory}
    receipts = []
    for path in (run / 'evaluations/unchanged-ir/compiler-receipts').glob('*.jsonl'):
        receipts.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    actual = {row['ast_sha256'] for row in receipts if row['override_applied']}
    assert expected <= actual, 'An expected target specialization never consumed the override'
    assert all(row['ir_sha256'] == row['original_ir_sha256'] for row in receipts if row['override_applied'])
    config = read(run / 'config.json')
    target = config['experiment_goal']['target_case']
    timings = [next(row for row in result['per_workload'] if row['uuid'] == target and row['phase'] == 'timing')
        for result in [baseline, noop]]
    record = dict(status='original_ir_override_verified', source_sha256=config['source_sha256'],
        expected_specializations=len(expected), observed_specializations=len(actual),
        baseline_target=timings[0], unchanged_ir_target=timings[1],
        baseline_over_noop=timings[0]['latency_ms'] / timings[1]['latency_ms'],
        receipt_count=len(receipts), compiler_identity=read(run / 'workspace/ir-inventory.json')['compiler_identity'],
        baseline_result=str(baseline_path), baseline_scope=baseline_scope,
        scope='Complete formal correctness and actual stage-override receipts; no optimization claim')
    save(run / 'original-ir-verification.json', record)
    save(run / 'workspace/baseline-result.json', dict(original=baseline, unchanged_ir=noop, verification=record))
    save(run / 'status.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--generate', action='store_true', help='Launch one model hypothesis after each verified no-op')
    args = parser.parse_args()
    root = args.campaign.resolve()
    marker = root / 'presentation-accepted.json'
    while not marker.exists():
        time.sleep(10)
    acceptance = read(marker)
    assert acceptance['operator_figures'] == 26 and acceptance['fixed_case_figures'] == 4
    assert acceptance['screenshot_review'] == 'passed' and acceptance['slides_saved']
    for name in ['batchnorm', 'qkv', 'upsample', 'scatter']:
        run = root / name
        try:
            captured = run / 'baseline-capture/manifest.json'
            if not captured.exists() or read(captured)['status'] != 'captured':
                capture(run)
            if not (run / 'workspace/ir-inventory.json').exists():
                seed(run)
            for invocation, baseline in [('frozen-baseline', True), ('unchanged-ir', False)]:
                if not (run / 'evaluations' / invocation / 'result.json').exists():
                    evaluate(run, invocation, baseline)
            record = verify_noop(run)
            print(json.dumps(dict(operator=name, status=record['status'],
                target_ratio=record['baseline_over_noop'])), flush=True)
            if args.generate and not (run / 'agent-calls/ir-001').exists():
                command = [sys.executable, str(root / 'tools/ir_model.py'), '--run', str(run),
                    '--invocation', 'ir-001', 'generate']
                with (run / 'generation-ir-001.log').open('x') as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, start_new_session=True)
                save(run / 'process-ir-001.json', dict(pid=process.pid, command=command,
                    source_sha256=read(run / 'config.json')['source_sha256'],
                    started_at=datetime.now(timezone.utc).isoformat()))
        except Exception as error:
            save(run / 'initialization-error.json', dict(error=type(error).__name__ + ': ' + str(error),
                timestamp=datetime.now(timezone.utc).isoformat()))
            print(json.dumps(dict(operator=name, status='requires_attention', error=str(error))), flush=True)
    save(root / 'initialization-status.json', dict(status='initialization_pass_finished',
        operator_status={name:read(root / name / 'status.json') for name in ['batchnorm','qkv','upsample','scatter']}))


if __name__ == '__main__':
    main()
