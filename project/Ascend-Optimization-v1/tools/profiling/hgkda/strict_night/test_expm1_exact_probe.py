"""原小张量补测不能代替新增8192实际路径的完整特殊值检查。"""
import unittest
from pathlib import Path
import tempfile
from expm1_exact_probe import expected_cases,qualified,file_sha


class ExactProbeTests(unittest.TestCase):
    def test_checker_identity_accepts_dunder_file_string(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checker.py'
            path.write_bytes(b'actual checker bytes')
            self.assertEqual(file_sha(str(path)),file_sha(path))

    def receipt(self):
        return dict(status='passed',source_sha256='actual-source',passed=12,total=12,
                    cases=[dict(case=name,passed=True,zero_sign_checked=True) for name in expected_cases()])

    def test_complete_matching_probe_is_qualified(self):
        self.assertTrue(qualified(self.receipt(),'actual-source'))

    def test_another_source_or_missing_path_is_not_qualified(self):
        value=self.receipt()
        self.assertFalse(qualified(value,'another-source'))
        value['cases'][0]['case']='small25-original-probe'
        self.assertFalse(qualified(value,'actual-source'))

    def test_passed_count_does_not_hide_missing_zero_check(self):
        value=self.receipt()
        value['cases'][0].pop('zero_sign_checked')
        self.assertFalse(qualified(value,'actual-source'))


if __name__=='__main__':unittest.main()
