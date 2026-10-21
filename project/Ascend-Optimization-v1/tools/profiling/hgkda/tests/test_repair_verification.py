"""不能仅凭汇总passed或部分语义检查把修复列为交付。"""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('repair_verification',Path(__file__).resolve().parents[1]/'verify-semantic-repairs.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class RepairVerificationTests(unittest.TestCase):
    def data(self):
        return {'status':'passed','operator':'value_selecting_reduction_backward','source_sha256':'source','checker_sha256':'checker',
                'physical_device':7,'logical_device':'npu:0','passed':48,'total':48,
                'cases':[{'dtype':dtype,'case':i,'status':'passed'} for dtype in ['torch.float16','torch.bfloat16','torch.float32'] for i in range(16)]}

    def test_complete_identity_and_cases_pass(self):
        self.assertTrue(m.semantic_identity(self.data(),'source','checker'))

    def test_moe_requires_its_own_complete_36_case_set(self):
        data=self.data();data.update(operator='moe_sum_reduce',passed=36,total=36)
        data['cases']=[r for r in data['cases'] if r['case']<12]
        self.assertTrue(m.semantic_identity(data,'source','checker','moe_sum_reduce'))
        self.assertFalse(m.semantic_identity(data,'source','checker'))

    def test_partial_list_with_passing_summary_is_rejected(self):
        data=self.data();data['cases'].pop()
        self.assertFalse(m.semantic_identity(data,'source','checker'))

    def test_duplicate_cases_cannot_fill_expected_count(self):
        data=self.data();data['cases'][-1]=data['cases'][0]
        self.assertFalse(m.semantic_identity(data,'source','checker'))

    def test_wrong_case_set_with_same_count_is_rejected(self):
        data=self.data();data['cases'][-1]['case']=100
        self.assertFalse(m.semantic_identity(data,'source','checker'))
        data=self.data();data['cases'][-1]['dtype']='torch.float64'
        self.assertFalse(m.semantic_identity(data,'source','checker'))

    def test_wrong_source_checker_or_card_cannot_reuse_pass(self):
        for key in ['source_sha256','checker_sha256','physical_device']:
            with self.subTest(key=key):
                data=self.data();data[key]='wrong'
                self.assertFalse(m.semantic_identity(data,'source','checker'))


if __name__=='__main__':unittest.main()
