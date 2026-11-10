"""远程控制只有在脚本完整执行后才成功，不以SSH零退出码替代完成证明。"""
import contextlib
import importlib.util
import io
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('remote_python', ROOT/'tools/maintenance/remote_python.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RemotePythonTests(unittest.TestCase):
    def invoke(self, result):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'check.py'
            source.write_text('print("result", end="")\n', encoding='utf8')
            with patch.object(module.subprocess, 'run', side_effect=result) as execute:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    code = module.main([str(source), '--host', 'test-host', '--python', '/env with spaces/bin/python'])
                return code, execute.call_args

    def test_zero_exit_without_completion_is_failure(self):
        code, _ = self.invoke(lambda *a, **k: subprocess.CompletedProcess([], 0, b'partial\n', b''))
        self.assertEqual(code, 2)
        code, _ = self.invoke(lambda *a, **k: subprocess.CompletedProcess([], 9, b'', b'failed\n'))
        self.assertEqual(code, 9)

    def test_quotes_interpreter_and_confirms_full_execution(self):
        def remote(command, **kwargs):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(compile(kwargs['input'], '<remote-test>', 'exec'), {})
            return subprocess.CompletedProcess(command, 0, output.getvalue().encode(), b'')
        code, call = self.invoke(remote)
        self.assertEqual(code, 0)
        self.assertEqual(shlex.split(call.args[0][-1]), ['/env with spaces/bin/python', '-'])
        self.assertFalse(call.kwargs.get('shell', False))


if __name__ == '__main__':
    unittest.main()
