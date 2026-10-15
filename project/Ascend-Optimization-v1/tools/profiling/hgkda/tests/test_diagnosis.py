"""验证阶段选择、缺失数据和指令热点定位，避免把采集存在当作因果结论。"""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('diagnose', ROOT / 'skills/hgkda/scripts/diagnose.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
sys.path.insert(0, str(ROOT.parent))
import hgkda_adapter as adapter


class DiagnosisTests(unittest.TestCase):
    def case(self, operator='rad2deg_', **metrics):
        return {'operator': operator, 'case_id': 'case-1', 'metrics': metrics,
                'profiles': {}, 'axes': {'dtype': 'float32'}}

    def test_simple_metrics_can_guide_one_experiment(self):
        result = d.diagnose_case(self.case(aiv_scalar_ratio=.8), levels=['metrics', 'instruction'])
        self.assertEqual(result['requested_collections'], [])
        self.assertEqual(result['hypotheses'][0]['category'], 'control')

    def test_missing_or_invalid_metrics_do_not_become_measured_zero(self):
        for metrics in [{}, {'aiv_vec_ratio': None}, {'aiv_vec_ratio': float('nan')},
                        {'aiv_vec_ratio': 95}]:
            result = d.diagnose_case(self.case(**metrics), levels=['metrics', 'instruction'])
            self.assertEqual(result['stage'], 'collect_metrics')
            self.assertEqual(result['hypotheses'], [])

    def test_overlap_requests_finer_evidence_without_adding_percentages(self):
        result = d.diagnose_case(self.case(aiv_mte2_ratio=.9, aiv_vec_ratio=.8),
                                levels=['metrics', 'instruction'])
        self.assertEqual(result['requested_collections'][0]['level'], 'instruction')
        self.assertEqual(len(result['hypotheses']), 2)
        self.assertEqual(result['roofline']['status'], 'not_established')

    def test_complex_path_does_not_assume_supported_simulator(self):
        case = self.case('scatter_add', aiv_scalar_ratio=.8)
        result = d.diagnose_case(case, levels=['metrics'])
        self.assertEqual(result['requested_collections'], [])
        self.assertTrue(result['limitations'])
        self.assertEqual(result['execution_path_status'], 'needs_reference_comparison')

    def test_instruction_artifacts_are_not_automatically_a_proven_bottleneck(self):
        case = self.case('scatter_add', aiv_mte2_ratio=.8)
        case['profiles']['instruction'] = {'artifacts': [{'path': 'instructions.json'}]}
        result = d.diagnose_case(case, levels=['metrics', 'instruction'])
        self.assertEqual(result['requested_collections'], [])
        self.assertEqual(result['stage'], 'instruction_evidence')
        self.assertTrue(result['limitations'])

    def test_hotspots_filter_input_preparation_kernels(self):
        listing = {'kernels': [
            {'name': 'random_input', 'cores': [{'instructions': [{'cycles': 99999}]}]},
            {'name': 'copy_kernel_abc', 'cores': [{'name': 'core0', 'instructions': [
                {'pc': '0x1', 'cycles': 5, 'running_time_us': .1},
                {'pc': '0x2', 'cycles': 90, 'running_time_us': 1.2}]}]}]}
        source = '@triton.jit\ndef copy_kernel(X):\n    pass\n'
        result = adapter.instruction_hotspots(listing, source, count=1)
        self.assertEqual(len(result['kernels']), 1)
        self.assertEqual(result['kernels'][0]['hotspots'][0]['pc'], '0x2')

    def test_real_historical_simulator_listing(self):
        project = ROOT.parents[2]
        path = project / ('operators/narrow_copy/reports/narrow-copy-combined-20261006/'
                          'profile-small-instruction-artifacts/a008-ascend-instructions.json')
        listing = json.loads(path.read_text(encoding='utf-8-sig'))
        result = adapter.instruction_hotspots(listing,
            '@triton.jit\ndef _narrow_copy_flat_kernel(X):\n    pass\n', count=5)
        self.assertEqual(result['status'], 'matched')
        self.assertTrue(result['kernels'][0]['hotspots'])
        self.assertLessEqual(len(result['kernels'][0]['hotspots']), 5)


if __name__ == '__main__':
    unittest.main()
