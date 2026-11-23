"""续接paired阶段失败的IR采集，保留原始日志及已完成侧。"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

from ir_campaign import read, save, sha
from ir_collect import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--kernel-name', required=True)
    parser.add_argument('--resume-id', required=True)
    parser.add_argument('--timeout-minutes', type=int, default=1)
    args = parser.parse_args()
    assert args.timeout_minutes > 0
    assert Path(args.resume_id).name == args.resume_id
    out = args.run.resolve() / 'observations' / args.invocation
    evaluation = args.run.resolve() / 'evaluations' / args.invocation
    assert out.is_dir() and not (out / 'status.json').exists()
    assert not (out / 'simulator').exists(), 'Existing simulator requires separate inspection'
    result = read(evaluation / 'result.json')
    assert result['status'] == 'PASSED' and result['num_passed'] == result['num_workloads']
    provenance = read(evaluation / 'provenance.json')
    implementation = evaluation / 'implementation'
    assert sha(implementation / 'frozen.py') == provenance['source_sha256']
    assert sha(implementation / 'ir-manifest.json') == provenance['manifest_sha256']
    for item in provenance['source_files']:
        assert sha(implementation / item['path']) == item['sha256']
    if provenance['changed_ir_count']:
        checks = read(evaluation / 'supplementary-summary.json')
        assert checks and all(item['status'] == 'passed' for item in checks.values())
        review = read(args.run / 'agent-calls' / args.invocation / 'ir-review.json')
        changed = {item['path']: item['sha256'] for item in read(implementation / 'ir-manifest.json')['overrides']
            if item['sha256'] != item['original_sha256']}
        assert review['decision'] == 'approved' and review['changed_ir'] == changed
    pair = out / 'paired'
    spec = read(pair / 'spec.json')
    assert spec['source_sha256'] == provenance['source_sha256']
    assert spec['ir_input']['original_manifest_sha256'] == provenance['manifest_sha256']
    preserved = {str(path): sha(path) for side in ['candidate', 'reference']
        if (pair / side / 'collection.json').exists()
        for path in sorted((pair / side).rglob('*')) if path.is_file()}
    record = out / 'resumes' / args.resume_id
    record.mkdir(parents=True, exist_ok=False)
    config = read(out / 'config.json')
    config.update(package=str(args.package.resolve()))
    save(record / 'config.json', config)
    save(record / 'provenance.json', dict(started_at=datetime.now(timezone.utc).isoformat(),
        original_config_sha256=sha(out / 'config.json'), original_pair_spec_sha256=sha(pair / 'spec.json'),
        evaluation_provenance_sha256=sha(evaluation / 'provenance.json'), preserved_files=preserved,
        reason='Resume incomplete paired acquisition with a separately frozen observation package',
        package_manifest_sha256=sha(args.package / 'method-manifest.json'), tool_sha256=sha(__file__)))
    environment = dict(os.environ, PYTHONPATH=str(args.package.resolve() / 'src'))

    def execute(name, command):
        with (record / (name + '.log')).open('x') as stream:
            process = subprocess.run([sys.executable, *command], env=environment,
                stdout=stream, stderr=subprocess.STDOUT)
        save(record / (name + '-process.json'), dict(command=command, returncode=process.returncode))
        return process.returncode

    assert execute('paired', ['-m', 'hgkda.runtime.collect_pair', '--config', str(record / 'config.json'),
        '--source', str(implementation / 'frozen.py'), '--ir-manifest', str(implementation / 'ir-manifest.json'),
        '--case', config['experiment_goal']['target_case'], '--output', str(pair), '--resume']) == 0
    assert all(sha(Path(path)) == digest for path, digest in preserved.items()), 'Completed side changed'
    discovery = read(pair / 'candidate-discovery.json')
    durations = {}
    for task in discovery['tasks']:
        if task['task_type'].startswith('KERNEL_AI') or task['task_type'] in {'KERNEL_MIX_AIV', 'KERNEL_MIX_AIC'}:
            durations[task['name']] = durations.get(task['name'], 0) + task['duration_us']
    assert args.kernel_name in durations
    save(out / 'simulator-selection.json', dict(kernel=args.kernel_name, observed_task_duration_us=durations,
        selection='explicit host choice after paired-stage recovery', resume=str(record)))
    simulator = out / 'simulator'
    returncode = execute('simulator', ['-m', 'hgkda.runtime.collect_simulator',
        '--config', str(record / 'config.json'), '--paired-directory', str(pair), '--output', str(simulator),
        '--kernel-name', args.kernel_name, '--soc', 'Ascend910B4',
        '--simulator-library', config['sdk'] + '/aarch64-linux/simulator/Ascend910B4/lib',
        '--timeout-minutes', str(args.timeout_minutes), '--trace-unit', 'us', '--allow-postcapture-exit'])
    summary = summarize(config, simulator)
    save(out / 'status.json', dict(status='ir_evidence_collected', source_sha256=config['source_sha256'],
        ir_manifest_sha256=provenance['manifest_sha256'], reports=len(summary['reports']),
        simulator_returncode=returncode, paired_recovery=str(record), preserved_files_verified=len(preserved)))


if __name__ == '__main__':
    main()
