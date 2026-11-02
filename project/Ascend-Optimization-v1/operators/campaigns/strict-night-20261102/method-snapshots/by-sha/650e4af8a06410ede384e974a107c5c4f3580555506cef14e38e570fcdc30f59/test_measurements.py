"""汇报分数必须由完整逐case延迟计算，不能忽略失败或使用算术平均。"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('night_measurements',Path(__file__).with_name('measurements.py'))
measurements=importlib.util.module_from_spec(spec);spec.loader.exec_module(measurements)


class ScoreTests(unittest.TestCase):
    def result(self):
        return dict(status='PASSED',is_hack=False,num_passed=3,num_workloads=3,geo_mean=1.,per_workload=[
            dict(uuid='correct',phase='correctness',status='PASSED'),
            dict(uuid='time1',phase='timing',status='PASSED',latency_ms=2.,reference_latency_ms=4.),
            dict(uuid='time2',phase='timing',status='PASSED',latency_ms=8.,reference_latency_ms=4.)])
    def test_geometric_mean_of_timing_only(self):self.assertEqual(measurements.score(self.result()),1.)
    def test_incorrect_baseline_cannot_block_slower_correct_repair(self):
        previous=dict(pytorch=dict(status='PASSED',score=2.),supplementary=dict(status='failed'))
        self.assertEqual(measurements.baseline_score(previous),0.)
        previous['supplementary']['status']='passed'
        self.assertEqual(measurements.baseline_score(previous),2.)
        previous['cann']=dict(score=.7)
        self.assertEqual(measurements.baseline_score(previous),.7)
    def test_wrong_aggregate_or_failed_case_rejected(self):
        result=self.result();result['geo_mean']=1.25
        with self.assertRaises(AssertionError):measurements.score(result)
    def test_whitespace_comments_docs_are_not_code_improvements(self):
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/'a.py';b=Path(folder)/'b.py'
            a.write_text('"""old docs"""\ndef run(x):\n    return x+1\n\n')
            b.write_text('# comment\n"""new docs"""\ndef run(x):\n    return x + 1\n')
            self.assertEqual(measurements.computation_identity(a),measurements.computation_identity(b))
            b.write_text('def run(x):\n    return x+2\n')
            self.assertNotEqual(measurements.computation_identity(a),measurements.computation_identity(b))
        result=self.result();result['per_workload'][0]['status']='FAILED'
        with self.assertRaises(AssertionError):measurements.score(result)


if __name__=='__main__':unittest.main()
