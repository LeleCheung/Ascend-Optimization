"""原语义数值修复不得冒充合格CANN性能交付。"""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from proposals import evaluation_plan
from report import build

controller_spec=importlib.util.spec_from_file_location('strict_proposal_test_controller',HERE/'run.py')
controller=importlib.util.module_from_spec(controller_spec)
controller_spec.loader.exec_module(controller)


class ProposalReferenceTests(unittest.TestCase):
    def test_original_only_keeps_pytorch_measurement_identity(self):
        plan=evaluation_plan('original_PyTorch_only')
        self.assertEqual(plan['folder'],'pytorch')
        self.assertEqual(plan['label'],'Codex-strict-PyTorch')
        self.assertEqual(plan['reference'],'original_PyTorch_only')

    def test_cann_measurement_names_do_not_change(self):
        plan=evaluation_plan('qualified_CANN')
        self.assertEqual(plan['folder'],'CANN')
        self.assertEqual(plan['label'],'Codex-strict-CANN')

    def test_unqualified_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluation_plan('unqualified_CANN')

    def test_fast_original_only_never_counts_as_cann_acceptance(self):
        rows=[dict(operator='op'+str(i),group='1015' if i<10 else '1029',accepted=False)
              for i in range(30)]
        best=dict(qualification='passed',compiled_status='verified',
                  review_status='reviewed_strict_triton',reference='original_PyTorch_only',
                  search_score=99.)
        result=build(dict(rows=rows),dict(status='running',operators={'op0':dict(best=best)}),
                     lambda row,candidate:row)
        self.assertEqual(result['accepted'],0)
        self.assertEqual(result['rows'][0]['selection'],'new_verified_candidate')
        self.assertIn('CANN参考不合格或缺失',result['rows'][0]['status'])

    def test_request_result_is_reused_without_new_device_command(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for leaf in ['catalog','original-catalog']:
                path=root/leaf/'ops/op'
                path.mkdir(parents=True)
                for name in ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']:
                    (path/name).write_text('same frozen contract')
            source=root/'main.py';source.write_text('def run(x): return x\n')
            formal=root/'Codex-strict-PyTorch-1.result.json'
            formal.write_text(json.dumps(dict(status='PASSED',num_passed=1,num_workloads=1,geo_mean=1.2)))
            formal.with_name('Codex-strict-PyTorch-1.request.json').write_text(json.dumps(dict(
                implementation=dict(sources=[dict(content=source.read_text())]))))
            (root/'extra-pytorch.json').write_text('{}')
            runner=SimpleNamespace(q=SimpleNamespace(records={'op':dict(reference='original_PyTorch_only')}),
                                   cmd=mock.Mock(side_effect=AssertionError('同源码不再次评测')))
            helper=SimpleNamespace(supplement_status=lambda *args:'passed')
            with mock.patch.object(controller,'BATCH',root), \
                    mock.patch.dict('sys.modules',validation_status=helper):
                result=controller.Runner.validate(runner,'op',source,root,existing_pytorch_result=formal)
            self.assertEqual(result['qualification'],'passed')
            self.assertEqual(set(result['extras']),{'pytorch'})
            runner.cmd.assert_not_called()


if __name__=='__main__':
    unittest.main()
