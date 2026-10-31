"""补采只能来自冻结输入和已通过当前准入的记录。"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from capture_record import choose_case,capture


class CaptureRecordTests(unittest.TestCase):
    def test_choose_largest_frozen_float32_without_mutating_input(self):
        small=dict(name='small',inputs=dict(spec=dict(dtype='float32',shape=[2,3])))
        large=dict(name='large',inputs=dict(spec=dict(dtype='float32',shape=[4,5])))
        half=dict(name='half',inputs=dict(spec=dict(dtype='float16',shape=[100,100])))
        cases=[small,half,large]
        self.assertIs(choose_case(cases),large)
        self.assertEqual(cases,[small,half,large])
        self.assertIs(choose_case([half]),half)
        with self.assertRaises(ValueError):choose_case([])

    @unittest.skipUnless(__import__('importlib.util',fromlist=['find_spec']).find_spec('fcntl'),'控制器运行于Linux')
    def test_incomplete_or_stale_record_rejected_before_device_work(self):
        runner=SimpleNamespace()
        for checked in [dict(status='pending'),dict(status='verified',correctness_passed=False),
                        dict(status='verified',correctness_passed=True,extra_status='requires_new_checks')]:
            with patch('report.verify',return_value=checked):
                with self.assertRaises(ValueError):capture(runner,{})


if __name__=='__main__':unittest.main()
