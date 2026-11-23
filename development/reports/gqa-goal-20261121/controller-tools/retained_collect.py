"""Collect current retained-source evidence without repeating full acceptance."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import subprocess
import sys

from ir_campaign import read, save, sha
from ir_collect import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--timeout-minutes', type=int, default=2)
    args = parser.parse_args()
    run = args.run.resolve()
    config = read(run / 'config.json')
    parent = read(run / 'retained-parent.json')
    source = run / 'workspace/candidate.py'
    assert sha(source) == parent['source_sha256']
    result = read(Path(parent['evaluation']))
    assert result['status'] == 'PASSED' and result['num_passed'] == result['num_workloads']
    assert sha(Path(parent['original_source'])) == sha(source)
    environment = dict(os.environ, PYTHONPATH=str(Path(config['package']) / 'src'))

    def execute(name, command):
        with (run / (name + '.log')).open('x') as log:
            process = subprocess.run([sys.executable, *command], env=environment,
                stdout=log, stderr=subprocess.STDOUT)
        save(run / (name + '-process.json'), dict(command=command, returncode=process.returncode))
        return process.returncode

    save(run / 'observation-status.json', dict(status='collecting_pair', source_sha256=sha(source)))
    pair = run / 'paired-parent'
    assert execute('paired-parent', ['-m', 'hgkda.runtime.collect_pair', '--config', str(run / 'config.json'),
        '--source', str(source), '--case', config['experiment_goal']['target_case'], '--output', str(pair)]) == 0
    discovery = read(pair / 'candidate-discovery.json')
    durations = defaultdict(float)
    for task in discovery['tasks']:
        if task['task_type'].startswith('KERNEL_AI') or task['task_type'] in {'KERNEL_MIX_AIV', 'KERNEL_MIX_AIC'}:
            durations[task['name']] += task['duration_us']
    assert durations
    kernel = max(durations, key=durations.get)
    save(run / 'simulator-selection.json', dict(kernel=kernel, observed_tasks=dict(durations),
        reason='Largest same-case AI-core task; all call tasks preserved in paired profiling'))
    save(run / 'observation-status.json', dict(status='collecting_simulator', source_sha256=sha(source), kernel=kernel))
    simulator = run / 'simulator-parent'
    execute('simulator-parent', ['-m', 'hgkda.runtime.collect_simulator', '--config', str(run / 'config.json'),
        '--paired-directory', str(pair), '--output', str(simulator), '--kernel-name', kernel,
        '--soc', 'Ascend910B4', '--simulator-library', config['sdk'] + '/aarch64-linux/simulator/Ascend910B4/lib',
        '--timeout-minutes', str(args.timeout_minutes), '--trace-unit', 'us', '--allow-postcapture-exit'])
    summary = summarize(config, simulator)
    save(run / 'observation-status.json', dict(status='parent_dual_evidence_ready', source_sha256=sha(source),
        case=config['experiment_goal']['target_case'], reports=len(summary['reports']),
        application_completed=summary['application_completed'],
        prior_complete_validation='Reused accepted identical source and fixed contract',
        scope='New profiling and simulator observations; original acceptance and timing remain frozen'))


if __name__ == '__main__':
    main()
