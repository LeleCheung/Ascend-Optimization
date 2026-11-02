"""诊断请求终态未知时不能静默重发，完整回执可以复用。"""
import tempfile
from pathlib import Path
import unittest

from execution_components import pending


class RequestTests(unittest.TestCase):
    def test_new_request_is_pending(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'result.json'
            self.assertEqual(pending({}, [path]), [path])

    def test_unknown_running_or_completed_request_is_not_resent(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'result.json'
            for status in ['running', 'completed']:
                with self.subTest(status=status), self.assertRaises(RuntimeError):
                    pending(dict(status=status), [path])

    def test_partial_receipt_does_not_allow_restarting_other_side(self):
        with tempfile.TemporaryDirectory() as temp:
            existing = Path(temp) / 'first.json'
            existing.write_text('{}')
            with self.assertRaises(RuntimeError):
                pending(dict(status='running'), [existing, Path(temp) / 'second.json'])

    def test_complete_receipts_are_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'result.json'
            path.write_text('{}')
            self.assertEqual(pending(dict(status='completed'), [path]), [])


if __name__ == '__main__':
    unittest.main()
