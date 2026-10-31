"""利用真实开发原件检查受益范围、模型输入和拒绝的流水方案。"""
import hashlib,json,math,statistics,unittest
from pathlib import Path
from templates import generate
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3]
BATCH=PROJECT/'operators/campaigns/matmul-pipeline-20261027'

class EvidenceTests(unittest.TestCase):
    def test_final_generator_preserves_executed_source_bytes(self):
        parent=PROJECT/'operators/campaigns/roofline-20261027/validation/delivered.py'
        for label in ['search-v2','search-v3']:
            plan=json.loads((BATCH/label/'plan.json').read_bytes())
            for config in plan['configs']:
                actual=(BATCH/label/config['id']/'candidate.py').read_bytes()
                self.assertEqual(generate(parent.read_text(encoding='utf8'),config).encode(),actual)

    def test_six_case_aggregation_does_not_hide_scope(self):
        data=json.loads((BATCH/'search-v2/state.json').read_bytes())
        row=next(r for r in data['cases'] if r['case']=='dual-m-reuse-b')
        self.assertEqual(len(row['measurements']),6)
        score=statistics.geometric_mean(v['speedup'] for v in row['measurements'])
        self.assertAlmostEqual(score,row['speedup'])
        self.assertTrue(any(v['speedup']<1 for v in row['measurements']))
        self.assertGreater(score,1)
        screened=[r for r in data['cases'] if r['status']=='screened']
        self.assertTrue(all(len(r['measurements'])==1 and 'speedup' not in r for r in screened))

    def test_reduced_pipeline_stage_is_not_assumed_improvement(self):
        data=json.loads((BATCH/'search-v3/state.json').read_bytes())
        row=next(r for r in data['cases'] if r['case']=='dual-n-stage1')
        self.assertEqual(row['status'],'passed')
        self.assertTrue(all(v['speedup']<1 for v in row['measurements']))
        self.assertLess(row['speedup'],.9)

    def test_formal_result_and_delivered_source_are_bound(self):
        data=json.loads((BATCH/'validation/comparison.json').read_bytes())
        self.assertEqual(len(data['cases']),15)
        score=statistics.geometric_mean(r['parent_ms']/r['candidate_ms'] for r in data['cases'])
        self.assertAlmostEqual(score,data['actual_candidate_over_parent'])
        self.assertGreater(score,1)
        self.assertEqual(hashlib.sha256((BATCH/'validation/delivered.py').read_bytes()).hexdigest(),data['source_sha256'])
        for side in ['parent','candidate']:
            result=json.loads((BATCH/'validation'/side/(side+'-1.result.json')).read_bytes())
            self.assertEqual(result['status'],'PASSED')
            self.assertEqual((result['num_passed'],result['num_workloads']),(42,42))

    def test_matched_confirmation_has_distinct_scope(self):
        data=json.loads((BATCH/'matched-confirmation/state.json').read_bytes())
        self.assertEqual(data['status'],'completed')
        self.assertEqual({(r['dtype'],r['size']) for r in data['cases']},{(d,s) for d in ['float16','bfloat16'] for s in [1024,4096]})
        self.assertAlmostEqual(statistics.geometric_mean(r['m_reuse_over_n_reuse'] for r in data['cases']),data['geomean'])
        self.assertTrue(all(r['m_reuse_over_n_reuse']>1 for r in data['cases']))
        selected=(BATCH/'search-v2/dual-m-reuse-b/candidate.py').read_bytes()
        self.assertEqual(hashlib.sha256(selected).hexdigest(),data['source_sha256']['dual-m-reuse-b'])

    def test_profile_sources_and_sizes_are_bound(self):
        data=json.loads((BATCH/'analysis/mechanism.json').read_bytes())
        self.assertEqual({r['size'] for r in data['pairs']},{1024,2048})
        for pair in data['pairs']:
            for side,row in pair['records'].items():
                source=(BATCH/'validation'/side/(side+'.py')).read_bytes()
                self.assertEqual(row['spec']['source_sha256'],hashlib.sha256(source).hexdigest())
                self.assertEqual(row['facts']['basic']['Device Id'],'7')

    def test_model_received_exact_frozen_history(self):
        root=BATCH/'agent-review';snapshot=(root/'search-snapshot.json').read_bytes()
        manifest=json.loads((root/'input-manifest.json').read_bytes())
        self.assertEqual(hashlib.sha256(snapshot).hexdigest(),manifest['search']['sha256'])
        self.assertIn(snapshot.decode(),(root/'prompt.md').read_text(encoding='utf8'))
        selected=json.loads((root/'proposal.json').read_bytes())['config_ids']
        tested=json.loads((BATCH/'search-v3/plan.json').read_bytes())['configs']
        self.assertEqual(selected,[c['id'] for c in tested])

if __name__=='__main__':unittest.main()
