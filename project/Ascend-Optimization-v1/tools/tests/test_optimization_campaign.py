"""验证模型输出、真实合同及故障状态保存，避免错误候选进入最佳版本。"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'tools/profiling'))
spec = importlib.util.spec_from_file_location('campaign', PROJECT / 'tools/profiling/optimization_campaign.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class CampaignTests(unittest.TestCase):
    def test_history_replay_profiles_and_repeats_without_model_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            source = project / 'known.py'
            source.write_text('def run(x): return x\n', encoding='utf-8')
            knowledge = project / 'knowledge.md'
            knowledge.write_text('historical method', encoding='utf-8')
            output = project / 'replay'
            result = output / 'seed-evaluation/seed-1.result.json'
            result.parent.mkdir(parents=True)
            result.with_name('seed-1.request.json').write_text(json.dumps({
                'implementation': {'sources': [{'content': source.read_text(encoding='utf-8')}]}}))
            argv = ['campaign', '--operator', 'logaddexp2', '--seed', str(source),
                    '--output', str(output), '--rounds', '1', '--knowledge', str(knowledge), '--replay']
            lock_api = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
            with patch.object(campaign, 'PROJECT', project), patch.object(sys, 'argv', argv), \
                    patch.dict(sys.modules, {'fcntl': lock_api}), \
                    patch.object(campaign.subprocess, 'run', return_value=SimpleNamespace(stdout='commit\n')) as git, \
                    patch.object(campaign, 'evaluate', return_value=result) as evaluate, \
                    patch.object(campaign, 'checked', return_value=({}, 1.267)), \
                    patch.object(campaign, 'profile_best', return_value={'cases': []}) as profile, \
                    patch.object(campaign, 'run') as generate:
                campaign.main()
                # 完成后的只读复用不应因归档提交改变HEAD而重新评测。
                git.side_effect = AssertionError('completed replay must not inspect the current HEAD')
                campaign.main()
            generate.assert_not_called()
            profile.assert_called_once()
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(evaluate.call_args.kwargs['repeats'], 2)
            state = json.loads((output / 'campaign-state.json').read_text(encoding='utf-8'))
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(state['history'][0]['status'], 'REPLAYED')
            self.assertEqual(state['final_source_sha256'], state['seed_sha256'])
            self.assertTrue((output / 'round-01/给CC的提示词.md').exists())

    def test_operator_knowledge_is_frozen_in_effective_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            common = project / 'common.md'
            common.write_text('common', encoding='utf-8')
            with patch.object(campaign, 'PROJECT', project):
                self.assertEqual(campaign.knowledge_for('amin', common), 'common')
                specific = project / 'operators/logaddexp2/优化知识.md'
                specific.parent.mkdir(parents=True)
                specific.write_text('pointwise', encoding='utf-8')
                combined = campaign.knowledge_for('logaddexp2', common)
                self.assertEqual(combined, 'common\n\npointwise')
                self.assertNotEqual(campaign.digest(combined), campaign.digest('common'))
                self.assertEqual(campaign.knowledge_for('amin', common), 'common')

    def test_offline_missing_evidence_does_not_submit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(campaign, 'call') as device_call, patch.object(campaign, 'run') as execute:
                with self.assertRaisesRegex(ValueError, '离线报告缺少'):
                    campaign.evaluate('amin', output / 'candidate.py', output,
                                      'trial', 'http://localhost', offline=True)
                device_call.assert_not_called()
                execute.assert_not_called()

    def test_completed_evaluations_reused_without_contacting_device(self):
        root = PROJECT / 'operators/matmul_bias_activation/reports/profiling-dual-acc-v9-20261012'
        with patch.object(campaign, 'call') as device_call:
            result = campaign.evaluate('matmul_bias_activation', root / 'optimized-v9.py',
                                       root, 'optimized-v9', 'http://localhost', repeats=2)
            self.assertEqual(result.name, 'optimized-v9-1.result.json')
            device_call.assert_not_called()

    def test_command_log_parent_created(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'nested/log.txt'
            campaign.run([sys.executable, '-c', 'print("ready")'], log)
            self.assertEqual(log.read_text().strip(), 'ready')

    def test_generation_environment_is_passed_to_child_without_global_change(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'generation.log'
            previous = os.environ.get('MAX_THINKING_TOKENS')
            env = os.environ.copy()
            env.update(MAX_THINKING_TOKENS='8192', CLAUDE_CODE_MAX_OUTPUT_TOKENS='16384')
            campaign.run([sys.executable, '-c',
                          'import os; print(os.environ["MAX_THINKING_TOKENS"], '
                          'os.environ["CLAUDE_CODE_MAX_OUTPUT_TOKENS"])'], log, env=env)
            self.assertEqual(log.read_text().strip(), '8192 16384')
            self.assertEqual(os.environ.get('MAX_THINKING_TOKENS'), previous)

    def test_uncertain_device_request_is_not_resubmitted(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'trial-1.request.json').write_text('{}')
            with patch.object(campaign, 'run') as execute:
                with self.assertRaises(campaign.InterruptedDeviceRequest):
                    campaign.evaluate('amin', output / 'candidate.py', output, 'trial', 'http://localhost')
                execute.assert_not_called()

    def test_other_same_card_service_busy_blocks_device_submission(self):
        free = {'scheduler': {'healthy': 1, 'broken': 0, 'active': 0, 'waiting': 0}}
        busy = {'scheduler': {'healthy': 1, 'broken': 0, 'active': 1, 'waiting': 0}}
        with patch.object(campaign, 'call', side_effect=[free, busy]):
            with self.assertRaises(campaign.InterruptedDeviceRequest):
                campaign.assert_card_idle()

    def test_claude_error_and_unsafe_code_rejected(self):
        with self.assertRaises(ValueError):
            campaign.parse_candidate(json.dumps({'is_error': True, 'result': 'error'}))
        for source in ('import os\ndef run(x): return x', 'from subprocess import run\ndef run(x): return x', 'def other(x): return x'):
            with self.assertRaises(ValueError):
                campaign.parse_candidate(json.dumps({'structured_output': {'source': source}}))

    def test_structured_output_and_plain_json_supported(self):
        proposal = {'source': 'def run(x): return x', 'hypothesis': '改变分块', 'expected_effect': '减少搬运'}
        for response in ({'structured_output': proposal}, {'result': json.dumps(proposal)}):
            self.assertEqual(campaign.parse_candidate(json.dumps(response)), proposal)
        response = 'model warning\n' + json.dumps({'type': 'assistant', 'message': {}}) + '\n' + json.dumps({
            'type': 'result', 'structured_output': proposal})
        self.assertEqual(campaign.parse_candidate(response), proposal)

    def test_operator_named_entry_gets_run_alias(self):
        proposal = {'source': 'def logaddexp2(x, y):\n    return x\n',
                    'hypothesis': '减少寻址', 'expected_effect': '摊薄固定开销'}
        parsed = campaign.parse_candidate(json.dumps({'result': json.dumps(proposal)}))
        self.assertIn('run = logaddexp2', parsed['source'])

    def test_existing_run_alias_is_preserved_and_undefined_alias_rejected(self):
        for source in ('def optimized(x): return x\nrun = optimized\n',
                       'def logaddexp2(x,y): return x\nrun = logaddexp2\n'):
            proposal = {'source': source, 'hypothesis': '减少寻址', 'expected_effect': '提速'}
            self.assertEqual(campaign.parse_candidate(json.dumps({'structured_output': proposal}))['source'], source)
        with self.assertRaisesRegex(ValueError, '缺少 run'):
            campaign.parse_candidate(json.dumps({'structured_output': {'source': 'run = missing'}}))

    def test_real_v9_contract_and_source_identity(self):
        root = PROJECT / 'operators/matmul_bias_activation/reports/profiling-dual-acc-v9-20261012'
        first, score = campaign.checked(root / 'optimized-v9-1.result.json')
        second, repeated = campaign.checked(root / 'optimized-v9-2.result.json', first)
        self.assertGreater(score, 0.8)
        self.assertGreater(repeated, 0.8)
        self.assertEqual(first['source_sha256'], second['source_sha256'])
        self.assertEqual(len(second['rows']), 42)

    def test_atomic_state_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'state.json'
            campaign.save(target, {'status': 'running'})
            campaign.save(target, {'status': 'completed'})
            self.assertEqual(json.loads(target.read_text())['status'], 'completed')
            self.assertEqual([p.name for p in target.parent.iterdir()], ['state.json'])

    def test_extract_actual_kernel_symbols(self):
        source = (PROJECT / 'operators/matmul_bias_activation/candidates/profiling-dual-acc-v9-20261012.py').read_text(encoding='utf-8')
        self.assertIn('mba_dual_dispatch_v8_kernel', campaign.prefixes(source))
        self.assertNotIn('run', campaign.prefixes(source))


if __name__ == '__main__':
    unittest.main()
