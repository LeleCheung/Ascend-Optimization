"""源码交付不能选语义失败版本，也不能受PyTorch参考漂移左右。"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('night_delivery', Path(__file__).resolve().parents[1]/'export-nightly-deliverables.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def measured(latency, reference=20., source='same'):
    return {'source_sha256': source, 'timing': {'case': {'latency_ms': latency, 'reference_latency_ms': reference}}}


class DeliveryTests(unittest.TestCase):
    def test_reference_drift_does_not_select_slower_candidate(self):
        baseline = measured(10.)
        faster = dict(status='verified', **m.scores(baseline, [measured(2., 2.), measured(2., 2.)]))
        slower = dict(status='verified', **m.scores(baseline, [measured(5., 100.), measured(5., 100.)]))
        self.assertLess(faster['relative_pytorch'][0], slower['relative_pytorch'][0])
        self.assertIs(m.choose([slower, faster]), faster)

    def test_invalid_fast_candidate_is_not_deliverable(self):
        valid = {'status': 'verified', 'selection_score': 1.2}
        self.assertIs(m.choose([{'status': 'invalid', 'selection_score': 100.}, valid]), valid)
        self.assertIsNone(m.choose([{'status': 'pending', 'selection_score': 100.}]))

    def test_case_set_and_source_identity_must_match(self):
        with self.assertRaisesRegex(ValueError, 'case不同'):
            m.scores(measured(10.), [measured(2.), {'source_sha256': 'same', 'timing': {}}])
        with self.assertRaisesRegex(ValueError, '源码不同'):
            m.scores(measured(10.), [measured(2.), measured(2., source='other')])

    def test_two_repeats_required(self):
        with self.assertRaisesRegex(ValueError, '两次完整复验'):
            m.scores(measured(10.), [measured(2.)])

    def test_existing_delivery_bytes_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'source/main.py'
            m.immutable(path, b'correct\n')
            m.immutable(path, b'correct\n')
            with self.assertRaisesRegex(ValueError, '不同字节'):
                m.immutable(path, b'different\n')
            self.assertEqual(path.read_bytes(), b'correct\n')


if __name__ == '__main__':
    unittest.main()
