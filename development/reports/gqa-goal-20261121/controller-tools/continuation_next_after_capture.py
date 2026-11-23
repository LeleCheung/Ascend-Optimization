"""Resume a hypothesis only after the owned postcapture controller has finished."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--captured-invocation', required=True)
    parser.add_argument('--capture-pid', type=int, required=True)
    parser.add_argument('--next-invocation', required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    proc = Path('/proc') / str(args.capture_pid)
    expected = ('continuation_capture_after_assessment.py', str(run), args.captured_invocation)
    while proc.exists():
        try:
            command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        except FileNotFoundError:
            break
        if not all(part in command for part in expected):
            raise RuntimeError('Postcapture PID identity changed; inspect before resuming')
        time.sleep(5)
    assessment = json.loads((run / 'assessments' / args.captured_invocation / 'result.json').read_bytes())
    log = run / ('postchange-' + args.captured_invocation + '.log')
    records = []
    for line in log.read_text().splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            records.append(value)
    if not any(v.get('status') == 'postchange_evidence_ready' and
               v.get('source_sha256') == assessment['source_sha256'] for v in records):
        raise RuntimeError('Postcapture requires inspection; workspace preserved')
    if (run / ('process-' + args.next_invocation + '.json')).exists():
        raise RuntimeError('Next invocation already exists; refusing duplicate launch')
    subprocess.run([sys.executable, str(run.parent / 'continuation_next.py'),
                    '--run', str(run), '--invocation', args.next_invocation], check=True)


if __name__ == '__main__':
    main()
