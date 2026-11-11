"""验证调度保留参数边界、失败状态，且帮助或错误动作不触发模型／设备操作。"""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('validation_dispatch', ROOT/'hgkda/src/hgkda/runtime/__main__.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ValidationDispatchTests(unittest.TestCase):
    def test_help_and_invalid_action_never_start_backend(self):
        with patch.object(module.subprocess, 'run') as execute:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(['--help']), 0)
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    module.main(['unknown-action'])
            execute.assert_not_called()

    def test_case_and_paths_remain_single_arguments_and_failure_is_returned(self):
        with patch.object(module.subprocess, 'run', return_value=subprocess.CompletedProcess([], 7)) as execute:
            result = module.main(['pair', '--config', 'run with spaces/config.json', '--case', 'case with spaces'])
            self.assertEqual(result, 7)
            command = execute.call_args.args[0]
            self.assertEqual(command[-4:], ['--config', 'run with spaces/config.json', '--case', 'case with spaces'])
            self.assertIn('pair', command)
            self.assertFalse(execute.call_args.kwargs.get('shell', False))


if __name__ == '__main__':
    unittest.main()
