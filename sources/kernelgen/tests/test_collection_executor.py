import base64
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import pytest

from kernelgen.agents.extractor.flaggems import collection_executor as executor, collection_worker as worker
from kernelgen.agents.extractor.flaggems.collection_config import ExtractionConfig, load_extraction_config, set_extraction_setting
from kernelgen.framework.worker_pool import get_max_workers, set_max_workers


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv('KERNELGEN_CLI_HOME', str(tmp_path / 'state'))
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    return ExtractionConfig(host='a100-extract', container='kernelgen-test', python=sys.executable)


def test_config_shares_file_and_lock_with_worker_settings(config):
    from kernelgen.cli.main import main
    endpoint = 'http://127.0.0.1:8000'
    set_max_workers(endpoint, 3)
    for key, value in config.model_dump().items():
        assert main(['config', 'set', f'extract.{key}', value]) == 0
    assert load_extraction_config() == config
    set_max_workers(endpoint, 2)
    assert load_extraction_config() == config
    assert get_max_workers(endpoint) == 2


def test_missing_config_is_not_local_fallback(config):
    with pytest.raises(ValueError, match='extract.host'):
        load_extraction_config()
    set_extraction_setting('host', config.host)
    with pytest.raises(ValueError, match='extract.container'):
        load_extraction_config()


@pytest.mark.parametrize('key,value', [('host', '-oProxyCommand=bad'), ('host', 'user@host'),
                                      ('container', 'name;bad'), ('python', 'python3'), ('python', '/bin/python\nbad')])
def test_reject_unsafe_configuration(config, key, value):
    with pytest.raises(ValueError):
        set_extraction_setting(key, value)


def test_ssh_uses_alias_and_quotes_container_command(config):
    config = config.model_copy(update={'python': '/opt/with space/python'})
    command = executor.ssh_command(config)
    assert command[-2] == 'a100-extract'
    assert shlex.split(command[-1])[:7] == ['docker', 'exec', '-i', config.container, config.python, '-u', '-c']
    assert shlex.split(command[-1])[-1] == Path(worker.__file__).read_text()
    assert 'BatchMode=yes' in command


def make_attempt(tmp_path, test_code):
    root = tmp_path / 'original'
    (root / 'src/sample').mkdir(parents=True)
    (root / 'src/sample/__init__.py').write_text('VALUE = "snapshot"\n')
    attempt = tmp_path / 'local attempt'
    attempt.mkdir()
    (attempt / 'collector.py').write_text('')
    (attempt / 'test_collect_cases.py').write_text(test_code)
    (attempt / 'request.json').write_text(json.dumps({'source_root': str(root)}))
    return root, attempt


SUCCESS = '''import json, os
from pathlib import Path
import sample
def test_collect():
    root = Path(json.loads(Path('request.json').read_text())['source_root'])
    assert Path(sample.__file__).is_relative_to(root)
    assert sample.VALUE == 'snapshot'
    assert 'ANTHROPIC_AUTH_TOKEN' not in os.environ
    Path('cases.json').write_text(json.dumps({'root': str(root)}))
    Path('executed-source.json').write_text('[]')
'''


def test_roundtrip_without_shared_paths_or_kg_install(config, tmp_path, monkeypatch):
    root, attempt = make_attempt(tmp_path, SUCCESS)
    monkeypatch.setenv('ANTHROPIC_AUTH_TOKEN', 'fixture-secret')
    monkeypatch.setattr(executor, 'ssh_command', lambda _: [sys.executable, '-u', worker.__file__])
    record = executor.execute_pytest(attempt, 5, config)
    assert json.loads((attempt / 'cases.json').read_text())['root'] != str(root)
    assert record['host'] == config.host
    assert record['execution'] == 'ssh_docker_pytest'
    assert Path(record['remote_workspace'], 'source/src/sample/__init__.py').is_file()
    assert (attempt / 'pytest.log').is_file()


def test_missing_ssh_is_infrastructure_failure(config, tmp_path, monkeypatch):
    _, attempt = make_attempt(tmp_path, SUCCESS)
    monkeypatch.setattr(executor, 'ssh_command', lambda _: ['/nonexistent/kg-test-ssh'])
    with pytest.raises(executor.CollectionTransportError, match='cannot start SSH'):
        executor.execute_pytest(attempt, 5, config)


def test_broken_pipe_on_close_does_not_mask_complete_result(config, tmp_path, monkeypatch):
    _, attempt = make_attempt(tmp_path, SUCCESS)
    monkeypatch.setattr(executor, 'ssh_command', lambda _: [sys.executable, '-u', worker.__file__])
    original_popen = subprocess.Popen
    class ClosingPipe:
        def __init__(self, pipe):
            self.pipe = pipe
        def write(self, data):
            return self.pipe.write(data)
        def flush(self):
            return self.pipe.flush()
        def close(self):
            self.pipe.close()
            raise BrokenPipeError('peer finished before buffered close')
    def spawn(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        process.stdin = ClosingPipe(process.stdin)
        return process
    monkeypatch.setattr(executor.subprocess, 'Popen', spawn)
    assert executor.execute_pytest(attempt, 5, config)['execution'] == 'ssh_docker_pytest'
    assert (attempt / 'cases.json').is_file()


@pytest.mark.parametrize('name', ['../escape', '/absolute'])
def test_worker_rejects_escaping_snapshot(tmp_path, name):
    with pytest.raises(ValueError, match='invalid snapshot path'):
        worker.write_files(tmp_path, {name: base64.b64encode(b'bad').decode()})


def test_snapshot_excludes_environment_files_and_rejects_external_links(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'env.sh').write_text('fixture')
    original = Path.read_bytes
    def read(path):
        assert path.name != 'env.sh', 'environment content must never be read'
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', read)
    assert worker.source_files(root) == {}
    (root / 'outside.py').symlink_to(tmp_path / 'outside.py')
    with pytest.raises(ValueError, match='escapes'):
        worker.source_files(root)


def test_ignored_cache_is_not_dirty_but_changed_source_is(tmp_path):
    root = tmp_path / 'checkout'
    root.mkdir()
    def git(*args):
        return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True)
    git('init')
    (root / '.gitignore').write_text('__pycache__/\n.pytest_cache/\n')
    (root / 'source.py').write_text('value = 1\n')
    git('add', '--', '.gitignore', 'source.py')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'fixture')
    before = executor.snapshot(root)[1:]
    (root / '__pycache__').mkdir()
    (root / '__pycache__/source.pyc').write_bytes(b'cache')
    assert executor.snapshot(root)[1:] == before
    (root / 'source.py').write_text('value = 2\n')
    with pytest.raises(ValueError, match='clean source checkout'):
        executor.snapshot(root)


def test_snapshot_preserves_relative_links_and_hashes_native_code(tmp_path):
    root, restored = tmp_path / 'source', tmp_path / 'restored'
    root.mkdir()
    restored.mkdir()
    (root / 'kernel.cpp').write_text('original native implementation')
    (root / 'alias.cpp').symlink_to('kernel.cpp')
    files = worker.source_files(root)
    worker.write_files(restored, files)
    assert (restored / 'alias.cpp').is_symlink()
    assert worker.snapshot_digest(worker.source_files(restored)) == worker.snapshot_digest(files)
    (restored / 'kernel.cpp').write_text('different native implementation')
    assert worker.snapshot_digest(worker.source_files(restored)) != worker.snapshot_digest(files)


def test_disconnect_kills_remote_pytest_group(config, tmp_path):
    _, attempt = make_attempt(tmp_path, 'import time\ndef test_wait():\n    time.sleep(60)\n')
    identity = json.loads((attempt / 'request.json').read_text())
    files, digest, _ = executor.snapshot(Path(identity['source_root']))
    request = {'files': files, 'snapshot_sha256': digest, 'identity': identity, 'timeout': 20,
               **{name: (attempt / name).read_text() for name in ('collector.py', 'test_collect_cases.py')}}
    process = subprocess.Popen([sys.executable, '-u', worker.__file__], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        process.stdin.write(json.dumps(request) + '\n')
        process.stdin.flush()
        ready = json.loads(process.stdout.readline())
        assert ready['event'] == 'ready'
        process.stdin.close()
        assert process.wait(timeout=10) == 0
        result = json.loads(process.stdout.read())
        assert result['error'] == 'connection closed'
        assert result['returncode'] < 0
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def test_cancel_is_not_model_retry(config, tmp_path, monkeypatch):
    from kernelgen.framework.run_control import WorkspaceRunControl, RunCancelled
    _, attempt = make_attempt(tmp_path, SUCCESS)
    control = WorkspaceRunControl(tmp_path / 'run')
    control.request_cancel('fixture cancellation')
    monkeypatch.setattr(executor, 'ssh_command', lambda _: pytest.fail('must not launch after cancel'))
    with pytest.raises(RunCancelled):
        executor.execute_pytest(attempt, 5, config, checkpoint=lambda: control.checkpoint('COLLECT'))


def test_cancellation_during_collection_closes_remote_input(config, tmp_path, monkeypatch):
    from kernelgen.framework.run_control import WorkspaceRunControl, RunCancelled
    _, attempt = make_attempt(tmp_path, 'import time\ndef test_wait():\n    time.sleep(60)\n')
    control = WorkspaceRunControl(tmp_path / 'run')
    # Simulate the SSH channel boundary: the worker is in a separate process
    # session and only observes a pipe EOF when the local transport is killed.
    relay = '''import subprocess, sys
p = subprocess.Popen([sys.executable, '-u', sys.argv[1]], stdin=subprocess.PIPE, start_new_session=True)
for line in sys.stdin.buffer:
    p.stdin.write(line)
    p.stdin.flush()
p.stdin.close()
p.wait()
'''
    monkeypatch.setattr(executor, 'ssh_command', lambda _: [sys.executable, '-u', '-c', relay, worker.__file__])
    def checkpoint():
        output = attempt / 'transport.jsonl'
        if output.exists() and 'ready' in output.read_text():
            control.request_cancel('during collection')
        control.checkpoint('COLLECT')
    with pytest.raises(RunCancelled):
        executor.execute_pytest(attempt, 20, config, checkpoint=checkpoint)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        lines = (attempt / 'transport.jsonl').read_text().splitlines()
        if len(lines) == 2:
            result = json.loads(lines[-1])
            assert result['error'] == 'connection closed'
            assert result['returncode'] < 0
            break
        time.sleep(0.05)
    else:
        pytest.fail('remote worker did not terminate after local cancellation')
