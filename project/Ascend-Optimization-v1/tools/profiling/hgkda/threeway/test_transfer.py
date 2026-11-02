"""传输必须确认远端Python实际完成，不能把堡垒机的退出码0当作同步成功。"""
import subprocess
import sys
import unittest
from unittest.mock import patch
import transfer


class TransportCompletion(unittest.TestCase):
    def test_asset_selection_error_with_zero_exit_is_rejected(self):
        response=subprocess.CompletedProcess([],0,'必须是唯一的资产 10.0.0.8'.encode('utf8'),b'')
        with patch.object(transfer.subprocess,'run',return_value=response):
            with self.assertRaisesRegex(RuntimeError,'未确认远端Python执行完成'):
                transfer.execute("print('never executed')")

    def test_remote_execution_preserves_exact_output_and_unicode(self):
        original=subprocess.run
        def local_python(command,**kwargs):
            self.assertEqual(command[-2:],['baai-910b','python3 -'])
            kwargs['input'].decode('ascii')
            kwargs['input']=b"import sys; sys.stdout.reconfigure(newline='\\n')\n"+kwargs['input']
            return original([sys.executable,'-'],**kwargs)
        with patch.object(transfer.subprocess,'run',side_effect=local_python):
            self.assertEqual(transfer.execute("print('中文结果')"),'中文结果\n'.encode('utf8'))

    def test_incomplete_remote_program_is_rejected(self):
        original=subprocess.run
        def local_python(command,**kwargs):return original([sys.executable,'-'],**kwargs)
        with patch.object(transfer.subprocess,'run',side_effect=local_python):
            with self.assertRaises(RuntimeError):transfer.execute("raise RuntimeError('stopped')")


if __name__=='__main__':unittest.main()
