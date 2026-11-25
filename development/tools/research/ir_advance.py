"""Advance reviewed and original IR validation serially, preserving failures while continuing other operators."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
import traceback

from ir_campaign import evaluate, read, save
from ir_initialize import verify_noop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--operators', nargs='+', default=['qkv', 'scatter', 'upsample'])
    parser.add_argument('--candidate', action='append', default=[])
    args = parser.parse_args()
    root = args.campaign.resolve()
    candidates = dict(item.split('=', 1) for item in args.candidate)
    log = root / 'advance-events.jsonl'

    def record(**event):
        event['timestamp'] = datetime.now(timezone.utc).isoformat()
        with log.open('a', encoding='utf8') as stream:
            import json
            stream.write(json.dumps(event) + '\n')
        print(event, flush=True)

    if args.wait_pid:
        path = Path('/proc') / str(args.wait_pid) / 'cmdline'
        while path.exists() and path.read_bytes():
            assert b'ir_campaign.py' in path.read_bytes(), 'Waiting process identity changed'
            time.sleep(3)

    for name in args.operators:
        run = root / name
        try:
            for invocation, baseline in [('frozen-baseline', True), ('unchanged-ir', False)]:
                output = run / 'evaluations' / invocation
                if not (output / 'result.json').exists():
                    evaluate(run, invocation, baseline=baseline, resume=(output / 'request.json').exists())
            verify_noop(run)
            record(operator=name, stage='original_ir_verified')
            invocation = candidates.get(name, 'unchanged-ir')
            if name in candidates:
                review = read(run / 'agent-calls' / invocation / 'ir-review.json')
                assert review['decision'] == 'approved'
                lowering = run / 'agent-calls' / invocation / 'lowering-review.json'
                assert read(lowering)['status'] == 'lowering_review_passed'
                evaluate(run, invocation)
                result = read(run / 'evaluations' / invocation / 'result.json')
                assert result['status'] == 'PASSED'
                checks = read(run / 'evaluations' / invocation / 'supplementary-summary.json')
                assert checks and all(row['status'] == 'passed' for row in checks.values())
            if name in candidates or name == 'scatter':
                command = [sys.executable, str(root / 'tools/ir_collect.py'), '--run', str(run),
                    '--invocation', invocation, '--package', str(root / 'observation-method/hgkda-v2')]
                if name == 'scatter':
                    command += ['--kernel-name', '_scatter_add_sort_out_kernel']
                with (run / ('advance-collection-' + invocation + '.log')).open('x') as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
                record(operator=name, stage='dual_collection_completed', invocation=invocation)
        except Exception as error:
            record(operator=name, stage='requires_attention', error=str(error), traceback=traceback.format_exc())
    save(root / 'advance-status.json', dict(status='staged_work_finished', events=str(log)))


if __name__ == '__main__':
    main()
