"""Run a local Python file using an explicit SSH host/interpreter without storing machine configuration."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys
import uuid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('script', type=Path, help='Local Python file; - reads standard input')
    parser.add_argument('--host', required=True, help='Configured SSH host or alias')
    parser.add_argument('--python', required=True, dest='interpreter', help='Remote Python interpreter path')
    parser.add_argument('--timeout', type=int, default=60)
    args = parser.parse_args(argv)
    if args.host.startswith('-') or args.timeout <= 0:
        parser.error('Host must not start with a hyphen and timeout must be positive')
    marker = '__HGKDA_REMOTE_COMPLETE_'+uuid.uuid4().hex+'__'
    raw = sys.stdin.buffer.read() if str(args.script) == '-' else args.script.read_bytes()
    payload = raw+('\nprint('+repr('\n'+marker)+')\n').encode('utf8')
    command = ['ssh', '-o', 'BatchMode=yes', args.host, shlex.join([args.interpreter, '-'])]
    result = subprocess.run(command, input=payload, capture_output=True, timeout=args.timeout)
    lines = result.stdout.decode('utf8', errors='replace').splitlines()
    for line in lines:
        if line != marker:
            print(line)
    if result.stderr:
        print(result.stderr.decode('utf8', errors='replace'), file=sys.stderr, end='')
    return result.returncode or (0 if marker in lines else 2)


if __name__ == '__main__':
    raise SystemExit(main())
