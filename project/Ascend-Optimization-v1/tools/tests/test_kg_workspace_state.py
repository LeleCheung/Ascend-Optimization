"""原生优化完成、worker退出与审核等待必须分别核实。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
from kg_workspace_state import exportable_state


class WorkspaceStateTests(unittest.TestCase):
    def test_waiting_review_needs_explicit_opt_in_and_finished_optimization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.kernelgen').mkdir()
            (root / '.kernelgen/run-progress.json').write_text(json.dumps({
                'state': 'PENDING', 'stage': 'WAITING_CODE_REVIEW'}))
            (root / '.kernelgen/run-process.json').write_text('{"finished_at":"2026-10-13T00:00:00Z"}')
            receipt = root / 'stages/optimize/attempts/01/result.json'
            receipt.parent.mkdir(parents=True)
            receipt.write_text('{"stage":"optimize","result":{"outcome":"COMPLETED"}}')
            with self.assertRaises(ValueError):
                exportable_state(root)
            self.assertEqual(exportable_state(root, True), 'OPTIMIZATION_COMPLETED_REVIEW_WAITING')
            (root / '.kernelgen/run-process.json').write_text('{}')
            with self.assertRaises(ValueError):
                exportable_state(root, True)
