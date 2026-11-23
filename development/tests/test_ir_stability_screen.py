"""Screen ABBA against both current-stage and historical retained measurements."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from development.tools.research.ir_wait_stability import screen_improvement, main


class StabilityScreenTest(unittest.TestCase):
    def test_conditional_policy_skips_abba_without_a_measurement_concern(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'validation-policy.json').write_text(json.dumps(dict(abba='conditional')))
            with patch('sys.argv', ['ir_wait_stability', '--run', str(run),
                    '--invocation', 'candidate', '--kgs-root', str(run)]), \
                    patch('development.tools.research.ir_wait_stability.subprocess.run') as execute:
                main()
            execute.assert_not_called()
            result = json.loads((run / 'stability-screen-candidate.json').read_bytes())
            self.assertEqual(result['status'], 'not_run_no_measurement_concern')

    def test_improvement_must_exceed_both_retained_measurements(self):
        for latency, expected in [(0.5, 'qualified_target_improvement'),
                                  (1.5, 'not_run_no_target_improvement')]:
            with self.subTest(latency=latency):
                root = Path('/campaign/operator')
                def result(value):
                    return dict(status='PASSED', num_passed=2, num_workloads=2,
                        per_workload=[dict(uuid='target', phase='timing', latency_ms=value)])
                documents = {
                    root / 'config.json': dict(experiment_goal=dict(target_case='target'),
                        continuation_parent='/prior/operator', continuation_invocation='retained'),
                    root / 'evaluations/candidate/result.json': result(latency),
                    root / 'evaluations/unchanged-ir/result.json': result(2),
                    Path('/prior/operator/evaluations/retained/result.json'): result(1)}
                with patch('development.tools.research.ir_wait_stability.read', side_effect=documents.__getitem__):
                    screened = screen_improvement(root, 'candidate', dict(supplement=dict(status='passed')))
                self.assertEqual(screened['status'], expected)

    def test_failed_supplement_prevents_performance_screening(self):
        result = dict(status='PASSED', num_passed=2, num_workloads=2)
        with patch('development.tools.research.ir_wait_stability.read', return_value=result) as read:
            screened = screen_improvement(Path('/campaign/operator'), 'candidate', dict(extra=dict(status='failed')))
        self.assertEqual(screened['status'], 'not_run_candidate_unqualified')
        self.assertEqual(read.call_count, 1)


if __name__ == '__main__':
    unittest.main()
