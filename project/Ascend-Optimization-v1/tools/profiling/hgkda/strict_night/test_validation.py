"""参考缺口算子在原PyTorch合同搜索后，只补正确性，不再次提交同一正式计时。"""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipIf(os.name=='nt','控制器依赖Linux卡锁；在910B宿主运行此测试')
class ValidationTests(unittest.TestCase):
    def test_existing_original_contract_result_is_reused(self):
        spec=importlib.util.spec_from_file_location('night_validation_controller',Path(__file__).with_name('run.py'))
        controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);op='cumulative_trapezoid'
            for catalog in ['catalog','original-catalog']:
                leaf=root/catalog/'ops'/op;leaf.mkdir(parents=True)
                for name in ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']:
                    (leaf/name).write_text('frozen')
            out=root/'candidate';out.mkdir();source=out/'main.py';source.write_text('def run(x): return x\n')
            (out/'extra-pytorch.json').write_text('{}')
            formal=root/'result.json'
            formal.write_text(json.dumps(dict(status='PASSED',is_hack=False,num_passed=1,num_workloads=1,geo_mean=.5)))
            formal.with_name('solution.json').write_text(json.dumps(dict(sources=[dict(content=source.read_text())])))
            runner=controller.Runner.__new__(controller.Runner)
            runner.q=SimpleNamespace(records={op:dict(reference='original_PyTorch_only')})
            runner.cmd=lambda *a,**kw:self.fail('同源码原合同已完成，不得再次提交计时')
            helper=SimpleNamespace(supplement_status=lambda *args:'passed')
            with patch.object(controller,'BATCH',root),patch.dict('sys.modules',validation_status=helper):
                value=runner.validate(op,source,out,existing_pytorch_result=formal)
                self.assertEqual(value['qualification'],'passed')
                self.assertEqual(value['pytorch_result'],str(formal))
                (root/'catalog/ops'/op/'oracle.py').write_text('changed')
                with self.assertRaises(AssertionError):runner.validate(op,source,out,existing_pytorch_result=formal)


if __name__=='__main__':unittest.main()
