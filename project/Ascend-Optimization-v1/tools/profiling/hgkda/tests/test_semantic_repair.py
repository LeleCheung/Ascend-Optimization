"""语义修复必须先通过同源码的完整补测，才允许发起性能复验。"""
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

spec=importlib.util.spec_from_file_location('semantic_repair',Path(__file__).resolve().parents[1]/'run-semantic-repair.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class SemanticRepairTests(unittest.TestCase):
    def moe_fixture(self,root):
        queue,source,old_output=self.fixture(root)
        op='moe_sum_reduce'
        queue.records={op:{'seed':'candidate.py'}}
        queue.plan={'execution_mode':'moe_semantic_repair','checker_sha256':hashlib.sha256(
            (queue.project/'tools/profiling/hgkda/check-nightly-semantics.py').read_bytes()).hexdigest()}
        output=queue.batch/'results'/op/'codex-stride-repair';output.mkdir(parents=True)
        data={'operator':op,'status':'passed','passed':36,'total':36,
              'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'checker_sha256':queue.plan['checker_sha256'],'physical_device':7,'logical_device':'npu:0',
              'cases':[{'dtype':dtype,'case':i,'status':'passed'}
                       for dtype in ['torch.float16','torch.bfloat16','torch.float32'] for i in range(12)]}
        path=output/'supplementary-semantics.json';path.write_text(json.dumps(data))
        return queue,source,output,path,data

    def test_moe_complete_semantics_allows_two_full_repeats(self):
        with TemporaryDirectory() as temp:
            queue,source,output,path,data=self.moe_fixture(Path(temp))
            self.assertTrue(m.execute(queue))
            queue.evaluate.assert_called_once_with('moe_sum_reduce',source,output/'independent','codex-stride-repair',2)

    def test_moe_wrong_case_set_is_rejected_even_with_passing_summary(self):
        with TemporaryDirectory() as temp:
            queue,source,output,path,data=self.moe_fixture(Path(temp))
            data['cases'][-1]['case']=200;path.write_text(json.dumps(data))
            with self.assertRaisesRegex(RuntimeError,'case集合'):m.execute(queue)
            queue.evaluate.assert_not_called()

    def dense_fixture(self):
        return {'status':'passed','source_sha256':'source','checker_sha256':'checker',
                'physical_device':7,'logical_device':'npu:0','passed':18,'total':18,
                'cases':[{'dtype':dtype,'sizes':sizes,'keepdim':keep,'status':'passed','repeats':5}
                         for dtype in ['torch.float16','torch.bfloat16','torch.float32']
                         for sizes in [[1,8209],[3,16387],[2,7,8209]] for keep in [False,True]]}

    def test_dense_tiles_require_all_cases_and_repeated_execution(self):
        data=self.dense_fixture()
        self.assertTrue(m.valid_dense_semantics(data,'source','checker'))
        data['cases'][0]['repeats']=1
        self.assertFalse(m.valid_dense_semantics(data,'source','checker'))

    def test_dense_tiles_cannot_hide_failed_case_with_passing_total(self):
        data=self.dense_fixture();data['passed']=17
        self.assertFalse(m.valid_dense_semantics(data,'source','checker'))
        data=self.dense_fixture();data['cases'][0]['status']='failed'
        self.assertFalse(m.valid_dense_semantics(data,'source','checker'))

    def fixture(self,root,**changes):
        op='value_selecting_reduction_backward';batch=root/'batch';batch.mkdir()
        source=batch/'candidate.py';source.write_bytes(b'candidate\n')
        project=root/'project';checker=project/'tools/profiling/hgkda/check-nightly-semantics.py'
        checker.parent.mkdir(parents=True);checker.write_bytes(b'checker\n')
        output=batch/'results'/op/'codex-stride-repair';output.mkdir(parents=True)
        result={'status':'passed','passed':48,'total':48,
                'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'checker_sha256':hashlib.sha256(checker.read_bytes()).hexdigest()}
        result.update(changes);(output/'supplementary-semantics.json').write_text(json.dumps(result))
        queue=SimpleNamespace(plan={'execution_mode':'semantic_repair'},records={op:{'seed':'candidate.py'}},
            batch=batch,project=project,state={'jobs':{}},path=batch/'queue-state.json',py='python',
            idle=Mock(),run=Mock(),evaluate=Mock(return_value={'status':'passed'}))
        def job(key,function):queue.state['jobs'][key]=function()
        queue.job=job
        return queue,source,output

    def test_complete_matching_semantics_allows_two_full_repeats(self):
        with TemporaryDirectory() as temp:
            queue,source,output=self.fixture(Path(temp))
            self.assertTrue(m.execute(queue))
            queue.evaluate.assert_called_once_with('value_selecting_reduction_backward',source,
                output/'independent','codex-stride-repair',2)
            queue.run.assert_not_called()

    def test_stale_source_cannot_reuse_passing_semantics(self):
        with TemporaryDirectory() as temp:
            queue,_,_=self.fixture(Path(temp),source_sha256='old-source')
            with self.assertRaisesRegex(RuntimeError,'身份不同'):m.execute(queue)
            queue.evaluate.assert_not_called()

    def test_partial_semantics_does_not_dispatch_performance(self):
        with TemporaryDirectory() as temp:
            queue,_,_=self.fixture(Path(temp),passed=47)
            with self.assertRaisesRegex(RuntimeError,'48项'):m.execute(queue)
            queue.evaluate.assert_not_called()

    def test_failure_is_preserved_and_not_overwritten(self):
        with TemporaryDirectory() as temp:
            queue,_,_=self.fixture(Path(temp),status='failed',passed=47)
            with self.assertRaisesRegex(RuntimeError,'已有失败补测'):m.execute(queue)
            queue.evaluate.assert_not_called();queue.run.assert_not_called()


if __name__=='__main__':unittest.main()
