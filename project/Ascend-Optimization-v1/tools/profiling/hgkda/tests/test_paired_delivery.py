"""交付不能只看汇总分数：核对原地语义、上传内容和逐case完整性。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

script = Path(__file__).resolve().parents[1]/'export-paired-deliverables.py'
spec = importlib.util.spec_from_file_location('paired_delivery', script)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class DeliveryTest(unittest.TestCase):
    def fixture(self, batch):
        source = batch/'candidates/log10_-rawptr-v1.py'
        source.parent.mkdir()
        source.write_bytes(b'def run(A):\r\n    return A\r\n')
        directory = batch/'results/log10_/codex-rawptr-v1'
        formal = directory/'formal'
        formal.mkdir(parents=True)
        rows = [dict(dtype=dtype, case=case, status='passed', repetitions=2)
                for dtype in ['torch.float16','torch.bfloat16','torch.float32'] for case in range(14)]
        (directory/'semantics.json').write_text(json.dumps(dict(status='passed', operator='log10_',
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), passed=42,total=42,
            checker_sha256=hashlib.sha256(script.with_name('check-pointwise-semantics.py').read_bytes()).hexdigest(),
            physical_device=7,logical_device='npu:0',cases=rows)))
        contract = batch/'catalog/ops/log10_'
        contract.mkdir(parents=True)
        (contract/'correctness.jsonl').write_text('{"name":"correct"}\n')
        (contract/'timing.jsonl').write_text('{"name":"timing"}\n')
        request = dict(implementation=dict(sources=[dict(path='main.py',content=source.read_text())]))
        (formal/'test-1.request.json').write_text(json.dumps(request))
        result = dict(status='PASSED',num_passed=2,num_workloads=2,geo_mean=2.,per_workload=[
            dict(uuid='correct',phase='correctness',status='PASSED'),
            dict(uuid='timing',phase='timing',status='PASSED',reference_latency_ms=2.,latency_ms=1.)])
        (formal/'test-1.result.json').write_text(json.dumps(result))
        return directory, formal

    def test_normalized_request_keeps_distinct_raw_source_sha(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p)
            result=m.read_arm(p,'log10_','codex-rawptr-v1')
            self.assertEqual(result['geo_mean'],2.)
            self.assertNotEqual(result['source_sha256'],result['submitted_source_sha256'])

    def test_partial_semantics_or_foreign_request_rejected(self):
        for failure in ['semantic_count','request','result']:
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as d:
                p=Path(d);directory,formal=self.fixture(p)
                if failure=='semantic_count':
                    f=directory/'semantics.json';v=json.loads(f.read_text());v['passed']=41
                elif failure=='request':
                    f=formal/'test-1.request.json';v=json.loads(f.read_text());v['implementation']['sources'][0]['content']='foreign'
                else:
                    f=formal/'test-1.result.json';v=json.loads(f.read_text());v['per_workload'].pop()
                f.write_text(json.dumps(v))
                with self.assertRaises(ValueError):m.read_arm(p,'log10_','codex-rawptr-v1')

    def test_dense_gate_requires_independent_complete_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);directory,_=self.fixture(p)
            with self.assertRaisesRegex(ValueError, '连续对齐'):
                m.read_arm(p,'log10_','codex-rawptr-v1',require_dense=True)
            rows=[dict(dtype=dtype,shape=shape,status='passed',repetitions=2,stream_switch_checked=True)
                  for dtype in ['torch.float16','torch.bfloat16','torch.float32']
                  for shape in [[1024],[65536],[1048576],[4096,4096]]]
            dense=dict(operator='log10_',status='passed',passed=12,total=12,cases=rows,
                source_sha256=m.sha(p/'candidates/log10_-rawptr-v1.py'),physical_device=7,logical_device='npu:0',
                checker_sha256=m.sha(script.with_name('check-dense-pointwise.py')))
            path=directory/'dense-semantics.json';path.write_text(json.dumps(dense))
            self.assertEqual(m.read_arm(p,'log10_','codex-rawptr-v1',require_dense=True)['dense_cases'],12)
            rows[0]['stream_switch_checked']=False;path.write_text(json.dumps(dense))
            with self.assertRaisesRegex(ValueError, '连续对齐'):
                m.read_arm(p,'log10_','codex-rawptr-v1',require_dense=True)
            dense['status']='failed';path.write_text(json.dumps(dense))
            self.assertIsNone(m.read_arm(p,'log10_','codex-rawptr-v1',require_dense=True))


if __name__ == '__main__':
    unittest.main()
