"""验证全量回归不会接受遗漏的用例或不能由原始耗时重建的分数。"""
import copy
from pathlib import Path
import runpy
import tempfile
import unittest


audit = runpy.run_path(str(Path(__file__).resolve().parents[3]/'development/tools/regression/audit.py'))
inspect_formal = audit['inspect_formal']


class RegressionAuditTests(unittest.TestCase):
    def setUp(self):
        self.expected = [('c0','correctness'),('t0','timing'),('t1','timing')]
        self.result = dict(status='PASSED',num_workloads=3,num_passed=3,server_backend='npu',
                           is_hack=False,timing_skipped=False,geo_mean=1.,per_workload=[
            dict(uuid='c0',phase='correctness',status='PASSED'),
            dict(uuid='t0',phase='timing',status='PASSED',speedup=2.,latency_ms=1.,reference_latency_ms=2.),
            dict(uuid='t1',phase='timing',status='PASSED',speedup=.5,latency_ms=4.,reference_latency_ms=2.)])

    def test_exact_case_set_and_geometric_mean(self):
        value = inspect_formal(self.result,self.expected)
        self.assertTrue(value['full_passed'])
        self.assertEqual(value['geo_mean'],1.)
        # 算术平均为1.25，不能替代几何平均。
        wrong = copy.deepcopy(self.result)
        wrong['geo_mean'] = 1.25
        with self.assertRaises(AssertionError):inspect_formal(wrong,self.expected)

    def test_missing_duplicate_or_renamed_case_rejected(self):
        for cases in [self.result['per_workload'][:-1],self.result['per_workload']+[self.result['per_workload'][0]]]:
            wrong = copy.deepcopy(self.result)
            wrong['per_workload'] = cases
            with self.assertRaises(AssertionError):inspect_formal(wrong,self.expected)
        wrong = copy.deepcopy(self.result)
        wrong['per_workload'][1]['uuid'] = 'different'
        with self.assertRaises(AssertionError):inspect_formal(wrong,self.expected)

    def test_timing_identity_and_backend_rejected(self):
        wrong = copy.deepcopy(self.result)
        wrong['per_workload'][1]['reference_latency_ms'] = 3.
        with self.assertRaises(AssertionError):inspect_formal(wrong,self.expected)
        wrong = copy.deepcopy(self.result)
        wrong['server_backend'] = 'cpu'
        with self.assertRaises(AssertionError):inspect_formal(wrong,self.expected)
        wrong = copy.deepcopy(self.result)
        wrong['timing_skipped'] = True
        self.assertFalse(inspect_formal(wrong,self.expected)['full_passed'])

    def test_archive_mapping_keeps_original_and_rejects_escape(self):
        original = '/data/project/runs/fixed'
        job = {key:original+'/'+key for key in ['source','review','archive','request']}
        job['checks'] = dict(supplementary=dict(leaf=original+'/catalog',checker=original+'/checker.py'))
        frozen = copy.deepcopy(job)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapped = audit['relocate_job'](job,original,root)
            self.assertEqual(Path(mapped['source']),root/'source')
            self.assertEqual(job,frozen)
            for value in ['/data/project/other/source',original+'/../source']:
                invalid = copy.deepcopy(job)
                invalid['source'] = value
                with self.assertRaises((AssertionError,ValueError)):
                    audit['relocate_job'](invalid,original,root)

    def test_external_catalog_requires_explicit_dependency_mapping(self):
        original='/data/project/runs/new';dependency='/data/project/runs/fixed'
        job={key:original+'/'+key for key in ['source','review','archive','request']}
        job['checks']=dict(supplementary=dict(leaf=dependency+'/catalog',checker=dependency+'/checker.py'))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(ValueError):audit['relocate_job'](job,original,root)
            mapped=audit['relocate_job'](job,original,root,{dependency:root/'frozen'})
            self.assertEqual(Path(mapped['checks']['supplementary']['checker']),root/'frozen/checker.py')
            self.assertEqual(job['checks']['supplementary']['checker'],dependency+'/checker.py')


if __name__ == '__main__':
    unittest.main()
