"""防止诊断读错 Native 形状，或把其他候选、其他合同的输入混入上下文。"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('native_context', ROOT / 'native_context.py')
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
spec = importlib.util.spec_from_file_location('build_overlay', ROOT / 'build-kg-overlay.py')
overlay = importlib.util.module_from_spec(spec);spec.loader.exec_module(overlay)


def fixture():
    workloads = [{'name': 'opaque-case-1', 'inputs': {
        'input': {'shape': [8, 8, 7168], 'dtype': 'bfloat16', 'type': 'random'},
        'scale': {'type': 'scalar', 'value': -1.25}}, 'seed': 13, 'tolerance': {'rtol': .015}}]
    solution = {'sources': [{'path': 'main.py', 'content': 'def run(x): return x'}]}
    result = {'status': 'PASSED', 'per_workload': [
        {'uuid': 'opaque-case-1', 'phase': 'timing', 'latency_ms': .09, 'axes': {}}]}
    identity = {'adapter_kind': 'native', 'round_num': 2,
        'solution_sha256': m.canonical_sha256(solution), 'evaluation_fingerprint': 'fingerprint',
        'result_sha256': m.canonical_sha256(result), 'workload_uuids': ['opaque-case-1'],
        'workload_sha256': [m.canonical_sha256(workloads[0])],
        'profile_workload_uuids': ['opaque-case-1']}
    context = {k: identity[k] for k in ['round_num', 'solution_sha256', 'evaluation_fingerprint', 'profile_workload_uuids']}
    context['per_workload'] = deepcopy(result['per_workload'])
    return context, {'identity': identity, 'solution': solution, 'result': result, 'workloads': workloads}


class NativeContextTests(unittest.TestCase):
    def test_opaque_case_gets_actual_shape_and_scalar_without_mutating_measurement(self):
        context, snapshot = fixture();original = deepcopy(context)
        value = m.enrich_native_context(context, snapshot)
        spec = value['hgkda_workloads']['opaque-case-1']
        self.assertEqual(spec['inputs']['input']['shape'], [8, 8, 7168])
        self.assertEqual(spec['inputs']['scale']['value'], -1.25)
        self.assertEqual(value['per_workload'], original['per_workload'])
        spec['inputs']['input']['shape'][0] = 999
        self.assertEqual(snapshot['workloads'][0]['inputs']['input']['shape'][0], 8)
        self.assertEqual(context, original)

    def test_foreign_candidate_round_contract_or_measurement_is_rejected(self):
        for kind in ['candidate', 'round', 'fingerprint', 'workload', 'result', 'context_result', 'profile_set']:
            with self.subTest(kind=kind):
                context, snapshot = fixture()
                if kind == 'candidate':snapshot['solution']['sources'][0]['content'] = 'different'
                elif kind == 'round':context['round_num'] = 3
                elif kind == 'fingerprint':context['evaluation_fingerprint'] = 'different'
                elif kind == 'workload':snapshot['workloads'][0]['inputs']['input']['shape'][0] = 2
                elif kind == 'result':snapshot['result']['per_workload'][0]['latency_ms'] = 0
                elif kind == 'context_result':context['per_workload'][0]['latency_ms'] = 0
                else:context['profile_workload_uuids'] = []
                with self.assertRaises(ValueError):m.enrich_native_context(context, snapshot)

    def test_legacy_context_remains_unchanged(self):
        context, snapshot = fixture();snapshot['identity']['adapter_kind'] = 'flaggems'
        self.assertEqual(m.enrich_native_context(context, snapshot), context)

    def test_overlay_extension_is_opt_in_and_baseline_stays_identical(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);source = root / 'repo/sources/kernelgen'
            (source / '.kernelgen/agents').mkdir(parents=True);(source / 'tools').mkdir()
            original = b'def get_profile_context(*args): return {}\n'
            (source / 'tools/profile_round.py').write_bytes(original)
            for name in ['kernel-profile-analyzer', 'kernel-knowledge-profile-analyzer']:
                (source / '.kernelgen/agents' / (name+'.md')).write_text('base role', encoding='utf8')
            base = overlay.build(root / 'default', repo=root / 'repo')
            enhanced = overlay.build(root / 'enhanced', repo=root / 'repo', native_context=True)
            self.assertNotIn('native_context_extension', base)
            self.assertEqual((root / 'default/kernelgen/tools/profile_round.py').read_bytes(), original)
            self.assertEqual((source / 'tools/profile_round.py').read_bytes(), original)
            self.assertIn('native_context_extension', enhanced)
            self.assertTrue((root / 'enhanced/kernelgen/tools/profile_round.py').read_bytes().startswith(original))
            self.assertEqual((root / 'enhanced/kernelgen/tools/hgkda_native_context.py').read_bytes(),
                             (ROOT / 'native_context.py').read_bytes())
            evidence = root/'paired.json'
            evidence.write_text(json.dumps({'schema_version':'hgkda-paired-v1',
                'cases':[{'operator':'log10_', 'source_sha256':'seed', 'evidence':'pair/comparison.json'}]}), encoding='utf-8')
            paired = overlay.build(root/'paired', repo=root/'repo', paired_evidence=evidence)
            embedded = root/'paired/kernelgen/.kernelgen/skills/hgkda/paired-evidence.json'
            self.assertEqual(embedded.read_bytes(), evidence.read_bytes())
            self.assertEqual(paired['paired_evidence_extension']['sha256'],
                             __import__('hashlib').sha256(evidence.read_bytes()).hexdigest())
            self.assertEqual((source/'tools/profile_round.py').read_bytes(), original)


if __name__ == '__main__':unittest.main()
