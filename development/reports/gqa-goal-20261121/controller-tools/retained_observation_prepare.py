"""Prepare observations from accepted identical delivery; reuse its full result."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from ir_campaign import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator-directory', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--target-case', required=True)
    args = parser.parse_args()
    template = read(args.template / 'config.json')
    environment = {k: template[k] for k in ['container', 'device', 'device_python', 'sdk', 'lock', 'kgs_server']}
    if 'device_pythonpath' in template:
        environment['device_pythonpath'] = template['device_pythonpath']
    envfile = args.run.parent / (args.run.name + '-environment.json')
    save(envfile, environment)
    env = dict(os.environ, PYTHONPATH=str(Path(template['package']) / 'src'))
    subprocess.run([sys.executable, '-m', 'hgkda.runtime.prepare', '--operator-directory', str(args.operator_directory),
        '--package', template['package'], '--environment', str(envfile), '--run-directory', str(args.run)],
        env=env, check=True)
    config = read(args.run / 'config.json')
    config.update(capture_entrypoint='run', capture_warmup=0, context_kind='performance_continuation',
        optimization_policy='performance_first', tools=template['tools'],
        experiment_goal=dict(kind='simulator_guided_optimization', target_case=args.target_case, new_control_arm=False))
    save(args.run / 'config.json', config)
    source = args.run / 'workspace/candidate.py'
    original = args.operator_directory / 'versions/independent_strict/retained/kernel.py'
    result = args.operator_directory / 'evidence/retained/evaluation/result.json'
    assert sha(source) == sha(original)
    accepted = read(result)
    assert accepted['status'] == 'PASSED' and accepted['num_passed'] == accepted['num_workloads']
    save(args.run / 'retained-parent.json', dict(source_sha256=sha(source), original_source=str(original),
        evaluation=str(result), scope='Accepted identical source and fixed contract; current profiling and simulator follow'))
    shutil.copyfile(args.template / 'validation-policy.json', args.run / 'validation-policy.json')
    subprocess.run([sys.executable, str(Path(__file__).with_name('retained_collect.py')),
        '--run', str(args.run)], check=True)


if __name__ == '__main__':
    main()
