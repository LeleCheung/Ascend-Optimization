"""验证性能未达标会继续优化，并选择拉低总体成绩的小输入。"""
import copy
import json
from pathlib import Path
import unittest

from hgkda.feedback import performance_feedback


ROOT=Path(__file__).resolve().parents[1]


class FeedbackTests(unittest.TestCase):
    def fixture(self):
        result=json.loads((ROOT/'benchmarks/skill-validation/20261109-selu/formal-3cec1212c3b7/operators/selu_/evaluation/result.json').read_bytes())
        cases=[json.loads(line) for line in (ROOT/'benchmarks/regressions/20261109/fixed-25/operators/selu_/cann_supplementary-catalog/timing.jsonl').read_text().splitlines()]
        return result,cases

    def test_correctness_pass_is_not_optimization_stop(self):
        result,cases=self.fixture()
        report=performance_feedback(result,cases,checks_passed=True)
        self.assertEqual(report['decision'],'continue_optimization')
        self.assertFalse(report['accepted'])
        self.assertEqual(len(report['selected_cases']),6)
        self.assertEqual({r['elements'] for r in report['selected_cases'] if r['case'].endswith(('3','7','11'))},{4096*4096})
        self.assertLess(report['ranked_cases'][0]['elements'],4096*4096)

    def test_score_threshold_and_checks_both_required(self):
        result,cases=self.fixture()
        report=performance_feedback(result,cases,threshold=.1,checks_passed=False)
        self.assertEqual(report['decision'],'complete_supplementary')
        self.assertEqual(performance_feedback(result,cases,threshold=.1,checks_passed=True)['decision'],'target_met')

    def test_contract_mismatch_rejected(self):
        result,cases=self.fixture()
        with self.assertRaises(ValueError):performance_feedback(result,cases[:-1])
        failed=copy.deepcopy(result);failed['status']='RUNTIME_ERROR'
        self.assertEqual(performance_feedback(failed,cases)['decision'],'repair_correctness_or_runtime')


if __name__=='__main__':unittest.main()
