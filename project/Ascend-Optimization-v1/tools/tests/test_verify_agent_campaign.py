"""用真实新批次检查最终源码与选择历史；篡改状态须被拒绝。"""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('verify_campaign', PROJECT / 'tools/profiling/verify-agent-campaign.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class VerifyCampaignTests(unittest.TestCase):
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
