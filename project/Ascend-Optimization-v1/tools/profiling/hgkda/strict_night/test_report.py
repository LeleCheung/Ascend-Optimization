"""防止待审查、补测失败或混合参考的高分被计作新增交付。"""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('night_report',Path(__file__).with_name('report.py'))
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


class DeliveryTests(unittest.TestCase):
    def previous(self):
        return dict(rows=[dict(operator='op'+str(i),group='1015' if i<10 else '1029',
            accepted=i<16,cann=dict(score=1. if i<16 else .6),status='frozen') for i in range(30)])
    def best(self):
        return dict(qualification='passed',compiled_status='verified',review_status='reviewed_strict_triton',
                    reference='qualified_CANN',search_score=.9)
    def result(self,best):
        return report.build(self.previous(),dict(status='running',operators={'op20':dict(best=best)}),lambda row,best:row)
    def test_pending_high_score_does_not_count(self):
        best=self.best();best['compiled_status']='pending_selected_verification'
        result=self.result(best)
        self.assertEqual(result['accepted'],16)
        self.assertIn('pending_candidate',result['rows'][20])
    def test_failed_extra_does_not_count(self):
        best=self.best();best['qualification']='failed'
        self.assertEqual(self.result(best)['accepted'],16)
    def test_verified_candidate_counts_in_correct_group(self):
        result=self.result(self.best())
        self.assertEqual(result['accepted'],17)
        self.assertEqual(result['groups']['1029']['accepted'],7)
    def test_pytorch_only_and_rounded_threshold_do_not_count(self):
        best=self.best();best['reference']='original_PyTorch_only'
        self.assertEqual(self.result(best)['accepted'],16)
        best=self.best();best['search_score']=.7999999
        self.assertEqual(self.result(best)['accepted'],16)


if __name__=='__main__':unittest.main()
