"""新复现报告只依赖本批合同，提交后无需重新测量旧基线。"""
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'tools/profiling'))
spec = importlib.util.spec_from_file_location('finalizer', PROJECT / 'tools/profiling/finalize-agent-campaign.py')
finalizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finalizer)


class ReplayReportTests(unittest.TestCase):
    def test_report_does_not_require_old_master_or_kg_contracts(self):
        batch = '20261014f-replay'
        name = 'profiling-agent-' + batch
        original = PROJECT / 'operators/logaddexp2/reports' / name
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            target = project / 'operators/logaddexp2/reports' / name
            target.mkdir(parents=True)
            for folder in ('seed-evaluation', 'final-evaluation'):
                shutil.copytree(original / folder, target / folder)
            for filename in ('campaign-state.json', 'final.py'):
                shutil.copyfile(original / filename, target / filename)
            with patch.object(finalizer, 'PROJECT', project), patch.object(finalizer, 'evaluate') as evaluate:
                report = finalizer.replay_report('logaddexp2', batch)
            evaluate.assert_not_called()
            text = report.read_text(encoding='utf-8')
            self.assertIn('1.269', text)
            self.assertIn('1.261', text)
            self.assertIn('本批完整合同', text)
            self.assertEqual(report.parent, target)


if __name__ == '__main__':
    unittest.main()
