"""真实演示证据回归：模型工具选择、部分模拟器、热点映射与正式计时。"""
import hashlib,importlib.util,json,statistics,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3];BATCH=PROJECT/'operators/campaigns/agent-demo-20261027'

class DemoTests(unittest.TestCase):
    def test_actual_model_tool_choices_match_returned_files(self):
        route=json.loads((BATCH/'agent/route/decision.json').read_bytes())
        self.assertIn('read_source',route['tools'])
        self.assertIn('read_ir',route['tools'])
        self.assertIn('read_hotspots',route['tools'])
        for name in route['tools']:
            d=json.loads((BATCH/'agent/tool-results'/(name+'.json')).read_bytes())
            self.assertEqual(d['tool'],name)
            self.assertTrue(d['result'])

    def test_simulator_crash_is_not_hidden_from_model(self):
        d=json.loads((BATCH/'agent/tool-results/read_hotspots.json').read_bytes())['result']
        self.assertEqual(d['status'],'partial')
        self.assertIn(139,d['child_exit_codes'])
        self.assertEqual(d['instructions'],1087)
        self.assertEqual(d['mapped_instructions'],983)

    def test_hotspot_source_line_is_actual_candidate(self):
        d=json.loads((BATCH/'seed-instruction-v3/summary.json').read_bytes())
        lines=(BATCH/'seed.py').read_text(encoding='utf8').splitlines()
        found=0
        for core in d['hotspots']['kernels']:
            for row in core['hotspots']:
                source=row.get('source',{})
                if source.get('file','').endswith('/seed.py') and 'code' in source:
                    self.assertEqual(lines[source['line']-1].strip(),source['code']);found+=1
        self.assertGreater(found,0)

    def test_model_selected_actions_were_measured(self):
        p=json.loads((BATCH/'agent/proposal.json').read_bytes())
        d=json.loads((BATCH/'search/state.json').read_bytes())
        self.assertEqual(p['config_ids'],[r['case'] for r in d['cases']])
        self.assertTrue(all(r['status']=='passed' and len(r['measurements'])==6 for r in d['cases']))
        best=max(d['cases'],key=lambda r:r['speedup'])
        self.assertEqual(best['case'],'dual-m-reuse-b')

    def test_formal_score_uses_identical_case_latency(self):
        d=json.loads((BATCH/'validation/comparison.json').read_bytes())
        self.assertEqual(len(d['cases']),15)
        score=statistics.geometric_mean(r['seed_ms']/r['candidate_ms'] for r in d['cases'])
        self.assertAlmostEqual(score,d['actual_candidate_over_seed'])
        self.assertEqual(hashlib.sha256((BATCH/'validation/delivered.py').read_bytes()).hexdigest(),d['source_sha256'])
        for side in ['seed','candidate']:
            r=json.loads((BATCH/'validation'/side/(side+'-1.result.json')).read_bytes())
            self.assertEqual((r['status'],r['num_passed'],r['num_workloads']),('PASSED',42,42))

if __name__=='__main__':unittest.main()
