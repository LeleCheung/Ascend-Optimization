"""Explicit-command capture adapter; configuration defines execution scope and outputs stay outside the repository."""
import os
from pathlib import Path
import subprocess

from ..artifacts import emit, identity, read_json


def collect(config_path, output_directory):
    config = read_json(config_path)
    argv = config['argv']
    if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
        raise ValueError('argv must be a nonempty array of strings')
    if Path(argv[0]).name not in {'msprof', 'msprof.exe'} or len(argv) < 2 or argv[1] != 'op':
        raise ValueError('Capture entry point must be an explicit msprof op command')
    source = identity(config['source'], kind='candidate_source')
    if source['sha256'] != config['source_sha256']:
        raise ValueError('Capture configuration belongs to different source')
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if any(v.startswith('--output') for v in argv):
        raise ValueError('The adapter supplies output; configuration must not duplicate it')
    argv = argv + ['--output=' + str(output / 'raw')]
    emit({'argv': argv, 'config': identity(config_path), 'source': source}, output / 'request.json')
    env = dict(os.environ)
    env.update(config.get('environment', {}))
    with (output / 'stdout.txt').open('x', encoding='utf8') as stdout, (output / 'stderr.txt').open('x', encoding='utf8') as stderr:
        process = subprocess.run(argv, cwd=config.get('cwd'), env=env, stdout=stdout, stderr=stderr,
                                 timeout=config.get('timeout_seconds', 600), check=False)
    result = {'status': 'completed' if process.returncode == 0 else 'failed', 'returncode': process.returncode,
              'source_sha256': source['sha256'], 'raw_directory': str(output / 'raw'),
              'scope': 'Capture process status only; raw reports still require identity verification and parsing before evidence acceptance'}
    emit(result, output / 'collection-process.json')
    return result
