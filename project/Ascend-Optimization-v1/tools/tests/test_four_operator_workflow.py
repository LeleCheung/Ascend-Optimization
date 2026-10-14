"""验证四算子计时选择与真实源码绑定，避免DMA被误计为kernel。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'tools/profiling'))
import four_operator_workflow as queue
import optimization_campaign as campaign


class FourOperatorTests(unittest.TestCase):
    def test_registry_binds_actual_sources_and_narrow_device_tasks(self):
        items = queue.registry()
        self.assertEqual(len(items), 4)
        narrow = next(item for item in items if item['operator'] == 'narrow_copy')
        self.assertEqual((narrow['port'], narrow['timing_scope']), (19657, 'device_task'))
        self.assertEqual(narrow['full_tests'], 33)
        for item in items:
            self.assertNotEqual(item['best']['sha256'], item['method_seed']['sha256'])

    def test_device_task_scope_is_passed_to_evaluator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(campaign, 'assert_card_idle'), patch.object(campaign, 'run') as execute:
                campaign.evaluate('narrow_copy', root / 'candidate.py', root, 'test',
                                  'http://127.0.0.1:19657', timing_scope='device_task')
            command = execute.call_args.args[0]
            self.assertEqual(command[command.index('--timing-scope') + 1], 'device_task')

    def test_cached_kernel_result_cannot_be_relabelled_as_device_task(self):
        root = PROJECT / 'operators/matmul_bias_activation/reports/profiling-dual-acc-v9-20261012'
        with patch.object(campaign, 'call') as call:
            with self.assertRaisesRegex(ValueError, '计时范围不同'):
                campaign.evaluate('matmul_bias_activation', root / 'optimized-v9.py', root,
                                  'optimized-v9', 'http://localhost', repeats=2, timing_scope='device_task')
            call.assert_not_called()

    def test_narrow_prompt_allows_real_dma_and_forbids_view(self):
        prompt = campaign.prompt_for('narrow_copy', 'def run(): pass', {'cases': []}, [], 'copy')
        self.assertIn('aclrtMemcpyAsync', prompt)
        self.assertIn('不能返回view', prompt)


if __name__ == '__main__':
    unittest.main()
