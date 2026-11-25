"""Retain failed captures and recollect using a binary-verified simulator target alias."""
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
    args = parser.parse_args()
    out = args.run / 'observations' / args.invocation
    config = read(out / 'config.json')
    config.update(package=str(args.package), ir_capture_runtime=str(Path(__file__).with_name('ir_override_runtime.py')),
        simulator_target_alias=dict(expected_arch='Ascend910B4-1', observed_arch='Ascend910B4'))
    archive = out / 'simulator-history/target-alias-rejection'
    archive.mkdir(parents=True, exist_ok=False)
    (out / 'simulator').rename(archive / 'simulator')
    for name in ['simulator.log', 'simulator-process.json']:
        path = out / name
        if path.exists():
            path.rename(archive / name)
    save(out / 'simulator-retry-config.json', config)
    save(out / 'simulator-retry-method.json', dict(started_at=datetime.now(timezone.utc).isoformat(),
        previous_capture=str(archive), reason='Simulator and hardware target names differ; binary hashes matched',
        runtime_sha256=sha(Path(config['ir_capture_runtime'])),
        package_manifest_sha256=sha(args.package / 'method-manifest.json')))
    kernel = read(out / 'simulator-selection.json')['kernel']
    command = [sys.executable, '-m', 'hgkda.runtime.collect_simulator', '--config',
        str(out / 'simulator-retry-config.json'), '--paired-directory', str(out / 'paired'),
        '--output', str(out / 'simulator'), '--kernel-name', kernel, '--soc', 'Ascend910B4',
        '--simulator-library', config['sdk'] + '/aarch64-linux/simulator/Ascend910B4/lib',
        '--timeout-minutes', '1', '--trace-unit', 'us', '--allow-postcapture-exit']
    environment = dict(os.environ, PYTHONPATH=str(args.package / 'src'))
    with (out / 'simulator.log').open('x') as stream:
        process = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    save(out / 'simulator-process.json', dict(command=command, returncode=process.returncode))
    # Parse valid partial windows separately; retain original exit, clock failures, and completion status.
    summary = summarize(config, out / 'simulator')
    save(out / 'status.json', dict(status='ir_evidence_collected', source_sha256=config['source_sha256'],
        ir_manifest_sha256=read(out / 'input-provenance.json')['manifest_sha256'],
        reports=len(summary['reports']), simulator_returncode=process.returncode))


if __name__ == '__main__':
    main()
