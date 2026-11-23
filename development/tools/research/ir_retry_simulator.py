"""保留失败模拟器原件，显式登记原因后延长单次采集防挂死时限。"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

from ir_campaign import read, save, sha
from ir_collect import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--invocation', required=True)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--retry-id', required=True)
    p.add_argument('--reason', required=True)
    p.add_argument('--timeout-minutes', type=int, required=True)
    args = p.parse_args()
    assert args.timeout_minutes > 0 and Path(args.retry_id).name == args.retry_id
    out = args.run.resolve() / 'observations' / args.invocation
    assert not (out / 'status.json').exists(), 'Completed observation remains frozen'
    simulator = out / 'simulator'
    assert simulator.is_dir()
    request = read(simulator / 'request.json')
    config = read(out / 'config.json')
    assert request['source_sha256'] == config['source_sha256']
    selection = read(out / 'simulator-selection.json')
    assert request['kernel'] == selection['kernel']
    previous_spec = read(simulator / 'spec.json')
    launch_replay = previous_spec.get('kernel_replay', {}).get('original', {}).get('manifest')
    if launch_replay:
        assert Path(launch_replay).is_file()
    archive = out / 'simulator-history' / args.retry_id
    archive.mkdir(parents=True, exist_ok=False)
    record = out / 'simulator-retries' / args.retry_id
    record.mkdir(parents=True, exist_ok=False)
    save(record / 'provenance.json', dict(started_at=datetime.now(timezone.utc).isoformat(),
        reason=args.reason, previous_request_sha256=sha(simulator / 'request.json'),
        previous_execution=read(simulator / 'execution.json'), previous_location=str(archive / 'simulator'),
        paired_spec_sha256=sha(out / 'paired/spec.json'), tool_sha256=sha(__file__),
        package_manifest_sha256=sha(args.package / 'method-manifest.json')))
    simulator.rename(archive / 'simulator')
    config.update(package=str(args.package.resolve()))
    save(record / 'config.json', config)
    command = ['-m', 'hgkda.runtime.collect_simulator', '--config', str(record / 'config.json'),
        '--paired-directory', str(out / 'paired'), '--output', str(simulator),
        '--kernel-name', selection['kernel'], '--soc', 'Ascend910B4',
        '--simulator-library', config['sdk'] + '/aarch64-linux/simulator/Ascend910B4/lib',
        '--timeout-minutes', str(args.timeout_minutes), '--trace-unit', 'us', '--allow-postcapture-exit']
    if launch_replay:
        command += ['--kernel-replay', launch_replay]
    environment = dict(os.environ, PYTHONPATH=str(args.package.resolve() / 'src'))
    with (record / 'simulator.log').open('x') as stream:
        process = subprocess.run([sys.executable, *command], env=environment,
            stdout=stream, stderr=subprocess.STDOUT)
    save(record / 'simulator-process.json', dict(command=command, returncode=process.returncode))
    summary = summarize(config, simulator)
    save(out / 'status.json', dict(status='ir_evidence_collected', source_sha256=config['source_sha256'],
        ir_manifest_sha256=read(out / 'input-provenance.json')['manifest_sha256'],
        reports=len(summary['reports']), simulator_returncode=process.returncode, recovery=str(record)))


if __name__ == '__main__':
    main()
