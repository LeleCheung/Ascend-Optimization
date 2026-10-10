"""核查跨平台路径解析，以及未闭环时不生成最终总表。"""
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
TOOL = PROJECT / 'tools/evaluation/summarize-master-goal.py'
spec = importlib.util.spec_from_file_location('master_summary', TOOL)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class GoalSummaryTest(unittest.TestCase):
    def test_linux_and_windows_records_map_to_local_evidence(self):
        relative = 'operators/amin/reports/direct-axis-native-v5-20261010/direct-axis-native-v5-1.result.json'
        expected = (PROJECT / relative).resolve()
        for recorded in ('/data/hanle/experiment/project/' + relative,
                         'E:\\repo\\project\\' + relative.replace('/', '\\')):
            self.assertEqual(summary.result_path(PROJECT, 'amin', recorded), expected)
        self.assertTrue(expected.is_file())

    def test_other_operator_and_escape_paths_are_rejected(self):
        for recorded in ('operators/matmul_bias_activation/reports/result.json',
                         'operators/amin/reports/../../../../outside.json'):
            with self.assertRaises(ValueError):
                summary.result_path(PROJECT, 'amin', recorded)

    def test_missing_closure_cannot_write_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / '成果.md'
            result = subprocess.run([sys.executable, str(TOOL), str(root / 'incomplete-project'), str(output)],
                                    capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix('.json').exists())


if __name__ == '__main__':
    unittest.main()
