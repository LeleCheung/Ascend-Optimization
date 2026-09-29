"""Opt-in Docker or full SSH + Docker integration; no model or GPU computation."""

import json
import os
import shlex
import subprocess
import time

import pytest

from kernelgen.agents.extractor.flaggems import collection_executor as executor
from kernelgen.agents.extractor.flaggems.collection_config import load_extraction_config
from kernelgen.framework.run_control import WorkspaceRunControl, RunCancelled
from kernelgen.tests.test_collection_executor import make_attempt, SUCCESS


pytestmark = pytest.mark.skipif(not os.environ.get('KG_COLLECTION_TEST_CONTAINER'), reason='opt-in Docker integration')


@pytest.fixture
def docker_config(monkeypatch, tmp_path):
    from kernelgen.cli.main import main
    monkeypatch.setenv('KERNELGEN_CLI_HOME', str(tmp_path / 'cli-state'))
    settings = {'host': os.environ.get('KG_COLLECTION_TEST_SSH_HOST', 'fixture-ssh-not-used'),
                'container': os.environ['KG_COLLECTION_TEST_CONTAINER'],
                'python': os.environ['KG_COLLECTION_TEST_PYTHON']}
    for key, value in settings.items():
        assert main(['config', 'set', f'extract.{key}', value]) == 0
    config = load_extraction_config()
    if not os.environ.get('KG_COLLECTION_TEST_SSH_HOST'):
        command = shlex.split(executor.ssh_command(config)[-1])
        monkeypatch.setattr(executor, 'ssh_command', lambda _: command)
    return config


def test_actual_docker_roundtrip(docker_config, tmp_path):
    root, attempt = make_attempt(tmp_path, SUCCESS)
    record = executor.execute_pytest(attempt, 10, docker_config)
    assert record['python'] == docker_config.python
    assert json.loads((attempt / 'cases.json').read_text())['root'] != str(root)


def test_actual_docker_disconnect_cleanup(docker_config, tmp_path):
    _, attempt = make_attempt(tmp_path, 'import time\ndef test_wait():\n    time.sleep(60)\n')
    control = WorkspaceRunControl(tmp_path / 'run')
    def checkpoint():
        output = attempt / 'transport.jsonl'
        if output.exists() and 'ready' in output.read_text():
            control.request_cancel('Docker disconnect test')
        control.checkpoint('COLLECT')
    with pytest.raises(RunCancelled):
        executor.execute_pytest(attempt, 10, docker_config, checkpoint=checkpoint)
    record = json.loads((attempt / 'execution.json').read_text())
    path = record['remote_workspace'] + '/result.json'
    probe = 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); print(p.read_text() if p.exists() else "")'
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        command = ['docker', 'exec', docker_config.container, docker_config.python, '-c', probe, path]
        if os.environ.get('KG_COLLECTION_TEST_SSH_HOST'):
            command = ['ssh', '-T', '-o', 'BatchMode=yes', docker_config.host, shlex.join(command)]
        result = subprocess.run(command,
                                capture_output=True, text=True, check=True, timeout=5)
        if result.stdout.strip():
            receipt = json.loads(result.stdout)
            assert receipt['error'] == 'connection closed'
            assert receipt['returncode'] < 0
            break
        time.sleep(0.1)
    else:
        pytest.fail('Docker pytest did not stop after transport cancellation')


def test_actual_collection_timeout(docker_config, tmp_path):
    _, attempt = make_attempt(tmp_path, 'import time\ndef test_wait():\n    time.sleep(60)\n')
    with pytest.raises(RuntimeError, match='collection timed out'):
        executor.execute_pytest(attempt, 1, docker_config)
    result = json.loads((attempt / 'transport.jsonl').read_text().splitlines()[-1])
    assert result['returncode'] < 0
    assert (attempt / 'pytest.log').is_file()
