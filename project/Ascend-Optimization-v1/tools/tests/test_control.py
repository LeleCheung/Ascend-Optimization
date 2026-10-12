"""验证控制端传输的路径限制、冲突处理与远端备份。"""
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'control', Path(__file__).resolve().parents[1] / 'control/control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)
NAME = 'operators/amin/reports/master-closure-20261010/精简汇报.md'


def archive(rows):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as output:
        for name, content, kind in rows:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(content)
            output.addfile(info, io.BytesIO(content))
    return stream.getvalue()


class ControlTests(unittest.TestCase):
    def test_status_reads_canonical_batch_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress = root / 'runs/20261010/kg-runs/amin-profile-20261010/.kernelgen/run-progress.json'
            progress.parent.mkdir(parents=True)
            progress.write_text('{"state":"completed"}', encoding='utf-8')
            parameters = {'root': str(root), 'batch': '20261010', 'operators': [], 'ports': []}
            namespace = {'parameters': parameters}
            from contextlib import redirect_stdout
            output = io.StringIO()
            with patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0, '', '')):
                with redirect_stdout(output):
                    exec(control.REMOTE_STATUS, namespace)
            rows = json.loads(output.getvalue())['runs']
            self.assertEqual(rows[0]['workspace'], 'amin-profile-20261010')
            self.assertEqual(rows[0]['state'], 'completed')

    def test_download_reads_diagnostics_from_canonical_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'Ascend-Optimization/project/Ascend-Optimization-v1').mkdir(parents=True)
            diagnostic = root / 'runs/20261010/diagnostics/amin-probe-20261010'
            diagnostic.mkdir(parents=True)
            (diagnostic / 'response.json').write_bytes(b'{"status":"SUCCEEDED"}')
            parameters = {'root': str(root), 'batch': '20261010', 'operators': ['amin'], 'max_bytes': 1024}
            script = 'parameters = ' + repr(parameters) + '\n' + control.REMOTE_DOWNLOAD
            result = subprocess.run([sys.executable, '-'], input=script.encode(), capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:gz') as archive:
                self.assertEqual(archive.getnames(), [
                    'operators/amin/reports/diagnostics-20261010/amin-probe-20261010/response.json'])

    def test_gateway_zero_exit_is_not_success(self):
        result = subprocess.CompletedProcess([], 0, b'no available gateway\r\n', b'')
        with patch.object(control.subprocess, 'run', return_value=result):
            with self.assertRaisesRegex(RuntimeError, 'no available gateway'):
                control.remote_python('baai-910b', {}, 'print(1)')

    def test_exec_requires_remote_bash_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'check.sh'
            script.write_text('echo ready\n', encoding='utf-8')
            result = subprocess.CompletedProcess([], 0, b'no available gateway\r\n', b'')
            with patch.object(control.subprocess, 'run', return_value=result):
                with self.assertRaisesRegex(RuntimeError, 'Bash'):
                    control.execute_script('baai-910b', script)

    def test_ssh_options_cannot_be_injected(self):
        for host in ('-oProxyCommand=cmd', 'host;cmd', 'host name'):
            with self.assertRaises(ValueError):
                control.ssh_command(host, 'python3 -')

    def test_malicious_archives_are_rejected(self):
        for name, kind in (('../escape', tarfile.REGTYPE),
                           ('operators/other/reports/20261010/result.json', tarfile.REGTYPE),
                           (NAME, tarfile.SYMTYPE),
                           ('operators/amin/reports/old/result.json', tarfile.REGTYPE)):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    control.download_actions(archive([(name, b'x', kind)]),
                                             Path(directory), ['amin'], '20261010')

    def test_duplicate_entries_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            row = (NAME, b'x', tarfile.REGTYPE)
            with self.assertRaises(ValueError):
                control.download_actions(archive([row, row]), Path(directory), ['amin'], '20261010')

    def test_raw_conflict_aborts_before_any_write(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            raw = project / 'operators/amin/reports/master-20261010/test.result.json'
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b'original')
            rows = [(NAME, b'new', tarfile.REGTYPE),
                    (raw.relative_to(project).as_posix(), b'changed', tarfile.REGTYPE)]
            with self.assertRaises(ValueError):
                control.download_actions(archive(rows), project, ['amin'], '20261010', True)
            self.assertFalse((project / NAME).exists())
            self.assertEqual(raw.read_bytes(), b'original')

    def test_checkout_line_endings_require_committed_blob(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            target = project / NAME
            target.parent.mkdir(parents=True)
            target.write_bytes(b'old\r\n')
            data = archive([(NAME, b'old\n', tarfile.REGTYPE)])
            with patch.object(control, 'committed_checkout', return_value=False):
                with self.assertRaises(ValueError):
                    control.download_actions(data, project, ['amin'], '20261010')
            with patch.object(control, 'committed_checkout', return_value=True):
                actions, count = control.download_actions(data, project, ['amin'], '20261010')
            self.assertEqual(actions, [])
            self.assertEqual(count, 1)
            self.assertEqual(target.read_bytes(), b'old\r\n')

    def test_derived_overwrite_preserves_old_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / 'project'
            target = project / NAME
            target.parent.mkdir(parents=True)
            target.write_bytes(b'old')
            data = archive([(NAME, b'new', tarfile.REGTYPE)])
            with self.assertRaises(ValueError):
                control.download_actions(data, project, ['amin'], '20261010')
            actions, count = control.download_actions(data, project, ['amin'], '20261010', True)
            state = Path(directory) / 'backups'
            control.apply_download(actions, project, state)
            self.assertEqual(count, 1)
            self.assertEqual(target.read_bytes(), b'new')
            self.assertEqual(next(state.rglob('精简汇报.md')).read_bytes(), b'old')

    def test_changed_local_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            actions, _ = control.download_actions(
                archive([(NAME, b'new', tarfile.REGTYPE)]), project, ['amin'], '20261010')
            target = project / NAME
            target.parent.mkdir(parents=True)
            target.write_bytes(b'user edit')
            with self.assertRaises(ValueError):
                control.apply_download(actions, project, project / 'backups')
            self.assertEqual(target.read_bytes(), b'user edit')

    def test_remote_upload_checks_and_backs_up_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / 'Ascend-Optimization/project/Ascend-Optimization-v1'
            target = project / 'tools/example.txt'
            target.parent.mkdir(parents=True)
            target.write_bytes(b'old')
            import base64
            import hashlib
            parameters = {'root': str(root), 'files': [{'name': 'tools/example.txt',
                'data': base64.b64encode(b'new').decode(),
                'sha256': hashlib.sha256(b'new').hexdigest()}]}
            script = 'parameters = ' + repr(parameters) + '\n' + control.REMOTE_UPLOAD
            result = subprocess.run([sys.executable, '-'], input=script.encode(), capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual(json.loads(result.stdout)['updated'], 1)
            self.assertEqual(target.read_bytes(), b'new')
            self.assertEqual(next((root / 'script-backups').rglob('example.txt')).read_bytes(), b'old')


if __name__ == '__main__':
    unittest.main()
