"""out补测必须覆盖所有dtype/接口形式，且绑定同一份源码。"""
import copy
import unittest
from out_api_probe import DTYPES, MODES, qualified


class OutApiProbeTests(unittest.TestCase):
    def fixture(self):
        return dict(status='passed', source_sha256='candidate',
                    cases=[dict(dtype=dtype, mode=mode, passed=True)
                           for dtype in DTYPES for mode in MODES])

    def test_complete_same_source_passes(self):
        self.assertTrue(qualified(self.fixture(), 'candidate'))

    def test_missing_or_failed_case_and_different_source_fail(self):
        original = self.fixture()
        for modification in ['missing', 'failed', 'source', 'duplicate']:
            record = copy.deepcopy(original)
            if modification == 'missing':
                record['cases'].pop()
            elif modification == 'failed':
                record['cases'][0]['passed'] = False
            elif modification == 'source':
                record['source_sha256'] = 'different'
            else:
                record['cases'][-1] = record['cases'][0]
            self.assertFalse(qualified(record, 'candidate'), modification)


if __name__ == '__main__':
    unittest.main()
