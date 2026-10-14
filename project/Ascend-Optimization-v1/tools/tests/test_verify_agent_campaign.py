"""用真实新批次检查最终源码与选择历史；篡改状态须被拒绝。"""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'tools/profiling'))
spec = importlib.util.spec_from_file_location('verify_campaign', PROJECT / 'tools/profiling/verify-agent-campaign.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class VerifyCampaignTests(unittest.TestCase):
    def test_actual_failed_and_noop_amin_reviews_are_bound_to_source(self):
        root = PROJECT / 'operators/amin/reports/profiling-agent-20261014-four-direct-method'
        report = verifier.verify(root)
        self.assertEqual(report['verified_generated_sources'], 3)
        self.assertEqual(report['semantically_rejected_sources'], 1)
        self.assertEqual(report['full_tests'], 27)
        path = root/'round-03/source-review.json'
        read = verifier.read
        review = copy.deepcopy(read(path))
        review['implementation_sha256'] = '0'*64
        def changed(candidate):
            return review if candidate == path else read(candidate)
        with patch.object(verifier, 'read', side_effect=changed):
            with self.assertRaisesRegex(ValueError, '未改实现的独立审查'):
                verifier.verify(root)

    def test_known_semantic_error_cannot_be_delivered_despite_passed_benchmark(self):
        root = PROJECT / 'operators/amin/reports/profiling-agent-20261012-agent'
        state = verifier.read(root / 'campaign-state.json')
        final_round = next(row for row in state['history'] if row.get('source_sha256') == state['final_source_sha256'])
        review_path = root / f'round-{final_round["round"]:02d}' / 'source-review.json'
        exists, read = Path.exists, verifier.read
        review = {'candidate_sha256':state['final_source_sha256'], 'status':'REJECTED_SEMANTIC',
                  'reason':'独立审查确认输出覆盖不足'}
        def changed(path):
            return review if path == review_path else read(path)
        with patch.object(Path, 'exists', lambda path: path == review_path or exists(path)), \
                patch.object(verifier, 'read', side_effect=changed):
            with self.assertRaisesRegex(ValueError, '已确认语义错误'):
                verifier.verify(root)

    def test_failed_generated_candidates_also_require_model_provenance(self):
        root = PROJECT / 'operators/logaddexp2/reports/profiling-agent-20261014-four-direct-method'
        report = verifier.verify(root)
        self.assertEqual(report['verified_generated_sources'], 3)
        read = verifier.read
        state = copy.deepcopy(read(root / 'campaign-state.json'))
        self.assertEqual(state['history'][0]['status'], 'FAILED')
        state['history'][0]['source_sha256'] = '0' * 64
        def changed(path):
            return state if path.name == 'campaign-state.json' else read(path)
        with patch.object(verifier, 'read', side_effect=changed):
            with self.assertRaisesRegex(ValueError, '生成候选源码'):
                verifier.verify(root)

    def test_actual_completed_campaigns(self):
        for operator, count in (('matmul_bias_activation', 42), ('amin', 27)):
            root = PROJECT / 'operators' / operator / 'reports/profiling-agent-20261012-agent'
            report = verifier.verify(root)
            self.assertEqual(report['full_tests'], count)
            self.assertEqual(report['status'], 'VERIFIED')
            self.assertGreater(report['artifact_count'], 0)

    def test_false_acceptance_and_final_score_are_rejected(self):
        root = PROJECT / 'operators/amin/reports/profiling-agent-20261012-agent'
        read = verifier.read
        original = read(root / 'campaign-state.json')
        for mutation in ('acceptance', 'final_score', 'false_replay'):
            state = copy.deepcopy(original)
            if mutation == 'acceptance':
                state['history'][0]['accepted'] = False
            elif mutation == 'final_score':
                state['final_score'] = max(state['final_scores'])
            else:
                state['history'][0]['status'] = 'REPLAYED'
            def changed(path):
                return state if path.name == 'campaign-state.json' else read(path)
            with patch.object(verifier, 'read', side_effect=changed):
                with self.assertRaises(ValueError):
                    verifier.verify(root)
