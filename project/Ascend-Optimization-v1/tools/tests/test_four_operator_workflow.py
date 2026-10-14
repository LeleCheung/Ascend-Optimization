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
    def test_delivery_requires_every_named_operator_and_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / 'operators').mkdir()
            items = queue.registry()
            with patch.object(queue, 'PROJECT', project):
                with self.assertRaisesRegex(ValueError, 'narrow_copy/method: missing'):
                    queue.summarize(items, 'delivery', ('replay', 'method'), require_complete=True)
                self.assertFalse((project / 'operators/四算子闭环-delivery.json').exists())
                rows = queue.summarize(items, 'progress', ('replay', 'method'))
            self.assertEqual(len(rows), 8)
            self.assertEqual({row['status'] for row in rows}, {'missing'})

    def test_method_retry_selection_keeps_replay_and_other_methods_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / 'operators').mkdir()
            items = queue.registry()
            replay_root = project/'operators/logaddexp2/reports/profiling-agent-replay-original-replay'
            retry_root = project/'operators/logaddexp2/reports/profiling-agent-logadd-retry-method'
            for root in (replay_root, retry_root):
                root.mkdir(parents=True)
                (root/'campaign-state.json').write_text('{"status":"running"}')
            with patch.object(queue, 'PROJECT', project):
                rows = queue.summarize(items, 'original', ('replay', 'method'),
                                       replay_batch='replay-original', method_batches={'logaddexp2':'logadd-retry'})
            logadd = [row for row in rows if row['operator']=='logaddexp2']
            self.assertEqual([(r['mode'],r['batch'],r['status']) for r in logadd],
                             [('replay','replay-original','running'),('method','logadd-retry','running')])
            amin = next(r for r in rows if r['operator']=='amin' and r['mode']=='method')
            self.assertEqual(amin['batch'],'original')
            self.assertIn('logadd-retry', (project/'operators/四算子闭环-original.md').read_text(encoding='utf-8'))

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
