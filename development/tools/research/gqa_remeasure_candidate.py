"""Resolve a documented timing concern with the unchanged four-case GQA contract."""
import argparse
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import shlex
import subprocess

from ir_campaign import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('--reason', required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    iteration = run / 'iterations' / args.invocation
    source = run / 'agent-calls' / args.invocation / 'output-source.py'
    original = iteration / 'parent-validation/results.json'
    assert read(original)['source_sha256'] == sha(source)
    config = read(iteration / 'config.json')
    out = run / 'conditional-remeasurement' / args.invocation
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'protocol.json', dict(reason=args.reason,
        registered_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(source), original_result=str(original),
        original_result_sha256=sha(original), contract_sha256=sha(run / 'contract.json'),
        rule='One complete four-case measurement with same-round CANN. Preserve first result, '
             'reuse bound post observation, retain only after ordinary selection checks.'))
    shell = 'source ' + shlex.quote(config['sdk'] + '/set_env.sh') + '\n'
    for key, value in dict(ASCEND_RT_VISIBLE_DEVICES=str(config['device']), OMP_NUM_THREADS='1',
                          TRITON_CACHE_DIR=str(out / 'cache'), PYTHONDONTWRITEBYTECODE='1',
                          TRITON_DISABLE_LINE_INFO='false').items():
        shell += 'export ' + key + '=' + shlex.quote(value) + '\n'
    shell += 'exec ' + shlex.join([config['device_python'], str(iteration / 'tools/attention_bench.py'),
        '--contract', str(run / 'contract.json'), '--candidate', str(source),
        '--output', str(out / 'results.json')])
    with Path(config['lock']).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (out / 'execution.log').open('x') as stream:
            process = subprocess.run(['docker', 'exec', '-w', str(out), config['container'],
                                      'bash', '-c', shell], stdout=stream, stderr=subprocess.STDOUT,
                                     timeout=1200)
    save(out / 'execution.json', dict(returncode=process.returncode))
    assert process.returncode == 0
    result = read(out / 'results.json')
    assert result['source_sha256'] == sha(source)
    assert result['contract_sha256'] == sha(run / 'contract.json')
    print([(r['case_id'], r['status'], r.get('candidate_timing')) for r in result['cases']])


if __name__ == '__main__':
    main()
