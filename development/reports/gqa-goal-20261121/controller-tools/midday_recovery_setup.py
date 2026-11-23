"""Start a project-owned KGS and prepare fixed-delivery regression runs.

Run on the 910B host. Candidate bytes are copied by the public runtime;
this script does not generate or patch operator implementations.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request


def main():
    base = Path('/data/hanle/ascend-optimization')
    run = base / 'runs/midday-recovery-walltime-20261117'
    run.mkdir(exist_ok=False)
    old = base / 'runs/night-research-20261116'
    service = json.loads((old / 'service.json').read_bytes())
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', 19720)) != 0, 'port occupied'
    command = [part.replace(str(old), str(run)).replace('19718', '19720').replace('--timing profiler', '--timing walltime')
               for part in service['command']]
    log = (run / 'kgs.log').open('xb')
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    (run / 'service.json').write_text(json.dumps(dict(pid=process.pid, command=command), indent=2))
    env = json.loads((old / 'environment.json').read_bytes())
    env['kgs_server'] = 'http://127.0.0.1:19720'
    (run / 'environment.json').write_text(json.dumps(env, indent=2))
    for _ in range(60):
        if process.poll() is not None:
            raise RuntimeError('KGS exited; inspect project log')
        try:
            with urllib.request.urlopen(env['kgs_server'] + '/status', timeout=2) as response:
                status = response.read()
            assert json.loads(status)['timing'] == 'walltime', 'fixed delivery timing required'
            (run / 'service-status.json').write_bytes(status)
            break
        except OSError:
            time.sleep(2)
    else:
        raise RuntimeError('KGS startup timeout')
    python = str(base / 'runtime/hgkda-release-venv/bin/python')
    package = old / 'release-v6/hgkda'
    child_env = dict(os.environ, PYTHONPATH=str(package / 'src'))
    operators = ['rad2deg_', 'log10_', 'upsample_bilinear2d', '_conj_physical',
                 'hardshrink_backward', 'round_out', 'hardswish',
                 'batch_norm_with_update_functional']
    for operator in operators:
        target = run / operator
        subprocess.run([python, '-m', 'hgkda.runtime', 'prepare',
                        '--operator-directory', str(base / 'Ascend-Optimization/operators' / operator),
                        '--package', str(package), '--environment', str(run / 'environment.json'),
                        '--run-directory', str(target)], env=child_env, check=True)
    (run / 'preparation.json').write_text(json.dumps(dict(operators=operators, package=str(package)), indent=2))
    # Serial full-contract checks: distinguish current environment from historical scores.
    for operator in operators:
        target = run / operator
        source = target / 'workspace/candidate.py'
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        with (target / 'evaluation.log').open('xb') as output:
            result = subprocess.run([python, '-m', 'hgkda.runtime', 'evaluate',
                                     '--config', str(target / 'config.json')],
                                    env=child_env, stdout=output, stderr=subprocess.STDOUT)
        (target / 'evaluation-exit.json').write_text(json.dumps(dict(returncode=result.returncode, source_sha256=digest)))
        print(operator, result.returncode, flush=True)


if __name__ == '__main__':
    main()
