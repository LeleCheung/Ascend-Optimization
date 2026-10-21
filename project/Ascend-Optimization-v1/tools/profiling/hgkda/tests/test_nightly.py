"""验证夜间队列不重复未知设备请求，并拒绝跨平台路径和源码漂移。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('nightly',Path(__file__).resolve().parents[1]/'run-nightly.py')
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)


class NightlyTests(unittest.TestCase):
    def prepare(self,root):
        batch=root/'Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/test'
        leaf=batch/'catalog/ops/square';leaf.mkdir(parents=True)
        (leaf/'definition.json').write_text('{}')
        source=batch/'seed.py';source.write_bytes(b'def run(x): return x*x\n')
        plan={'server':'http://127.0.0.1:1','records':[{'operator':'square','seed':'seed.py',
            'catalog_leaf':'catalog/ops/square','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}]}
        (batch/'plan.json').write_text(json.dumps(plan))
        return batch,source,plan

    def test_unknown_request_stops_before_resubmission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);batch,source,_=self.prepare(root);queue=n.Queue('test',root,'unused')
            output=batch/'result';output.mkdir();(output/'candidate-1.request.json').write_text('{}')
            with self.assertRaises(n.Attention):
                queue.evaluate('square',source,output,'candidate',1)

    def test_completed_job_is_not_relaunched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.prepare(root);queue=n.Queue('test',root,'unused')
            queue.state['jobs']['baseline:square']={'status':'passed'}
            queue.job('baseline:square',lambda:self.fail('重复执行'))

    def test_adapter_repair_batch_only_runs_selected_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);batch,_,plan=self.prepare(root)
            plan['execution_mode']='baseline_only';(batch/'plan.json').write_text(json.dumps(plan))
            queue=n.Queue('test',root,'unused');queue.idle=lambda:None
            queue.baseline=lambda op:{'status':'passed','operator':op}
            queue.kg=lambda *args:self.fail('接口补测批次不得派发优化任务')
            queue.run=lambda *args:self.fail('接口补测无需创建HG覆盖层')
            queue.execute()
            self.assertEqual(queue.state['status'],'completed')
            self.assertEqual(list(queue.state['jobs']),['baseline:square'])

    def test_bad_paths_and_changed_source_fail_before_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);batch,source,plan=self.prepare(root)
            plan['records'][0]['seed']='seeds\\square.py'
            (batch/'plan.json').write_text(json.dumps(plan))
            with self.assertRaises(n.Attention):n.Queue('test',root,'unused')
            plan['records'][0]['seed']='seed.py';(batch/'plan.json').write_text(json.dumps(plan))
            source.write_bytes(b'def run(x): return x\n')
            with self.assertRaises(n.Attention):n.Queue('test',root,'unused')


if __name__=='__main__':unittest.main()
