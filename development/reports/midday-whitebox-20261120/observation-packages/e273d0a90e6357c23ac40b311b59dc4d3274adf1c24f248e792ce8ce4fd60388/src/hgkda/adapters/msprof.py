"""显式命令采集适配；执行范围由配置给出，输出不写入仓库。"""
import os
from pathlib import Path
import subprocess

from ..artifacts import emit, identity, read_json


def collect(config_path, output_directory):
    config = read_json(config_path)
    argv = config['argv']
    if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
        raise ValueError('argv 必须为非空字符串数组')
    if Path(argv[0]).name not in {'msprof', 'msprof.exe'} or len(argv) < 2 or argv[1] != 'op':
        raise ValueError('采集入口必须是显式 msprof op 命令')
    source = identity(config['source'], kind='candidate_source')
    if source['sha256'] != config['source_sha256']:
        raise ValueError('采集配置与源码不同')
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if any(v.startswith('--output') for v in argv):
        raise ValueError('output 由本适配器指定，配置中不得重复')
    argv = argv + ['--output=' + str(output / 'raw')]
    emit({'argv': argv, 'config': identity(config_path), 'source': source}, output / 'request.json')
    env = dict(os.environ)
    env.update(config.get('environment', {}))
    with (output / 'stdout.txt').open('x', encoding='utf8') as stdout, (output / 'stderr.txt').open('x', encoding='utf8') as stderr:
        process = subprocess.run(argv, cwd=config.get('cwd'), env=env, stdout=stdout, stderr=stderr,
                                 timeout=config.get('timeout_seconds', 600), check=False)
    result = {'status': 'completed' if process.returncode == 0 else 'failed', 'returncode': process.returncode,
              'source_sha256': source['sha256'], 'raw_directory': str(output / 'raw'),
              'scope': '采集进程状态；原始报告仍需身份校验和解析，不等同完整证据已通过'}
    emit(result, output / 'collection-process.json')
    return result
