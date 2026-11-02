"""在独立tmux会话运行队列；控制端断开不影响执行。"""
import argparse
import shlex
import subprocess
from common import BATCH, HERE, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['all', 'references', 'optimize', 'priority', 'search', 'extra'], default='all')
    a = parser.parse_args()
    session = 'hanle-cann-priority-20261102-' + a.phase
    if subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0:
        print('EXISTS', session)
        return
    runtime = ROOT / 'runs' / BATCH.name
    runtime.mkdir(parents=True, exist_ok=True)
    command = ' '.join(shlex.quote(str(s)) for s in [ROOT / 'runtime/kg-controller/venv/bin/python', '-u', HERE / 'run.py', '--phase', a.phase])
    command += ' >> ' + shlex.quote(str(runtime / ('host-' + a.phase + '.log'))) + ' 2>&1'
    script = runtime / ('launch-' + a.phase + '.sh')
    script.write_text('#!/bin/bash\nset -euo pipefail\n' + command + '\n', encoding='utf8')
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, 'bash', str(script)], check=True)
    print('STARTED', session, runtime)


if __name__ == '__main__':
    main()
