"""检查诊断不会把参考变慢、错误源码或其他合同当成优化证据。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'skills/hgkda/scripts/analyze_execution_paths.py'
s=importlib.util.spec_from_file_location('execution_diagnosis',SCRIPT)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.source=self.root/'main.py';self.source.write_text('def run(x): return x\n')
        self.catalog=self.root/'catalog';self.catalog.mkdir()
        self.cases=[dict(name='example-timing-0',inputs=dict(spec=dict(shape=[1024],dtype='float32'))),
            dict(name='example-timing-1',inputs=dict(spec=dict(shape=[1048576],dtype='float32')))]
        (self.catalog/'definition.json').write_text(json.dumps(dict(name='example')))
        (self.catalog/'oracle.py').write_text('def run(x): return x\n')
        (self.catalog/'correctness.jsonl').write_text('{}\n')
        (self.catalog/'timing.jsonl').write_text('\n'.join(json.dumps(c) for c in self.cases)+'\n')
    def tearDown(self):self.tmp.cleanup()
    def result(self,name,reference,candidate):
        p=self.root/(name+'.result.json')
        v=dict(status='PASSED',num_passed=2,num_workloads=2,per_workload=[dict(uuid=c['name'],phase='timing',status='PASSED',latency_ms=t,reference_latency_ms=r) for c,r,t in zip(self.cases,reference,candidate)])
        p.write_text(json.dumps(v))
        p.with_name(name+'.request.json').write_text(json.dumps(dict(binding=dict(bundle_id='frozen'),settings=dict(num_trials=3),implementation=dict(sources=[dict(content=self.source.read_text())]))))
        return p
    def test_reference_drift_is_separate(self):
        old=self.result('old',[1,1],[2,2]);new=self.result('new',[2,2],[2,2])
        v=m.analyze(new,self.source,self.catalog,parent=old)
        self.assertAlmostEqual(v['score'],1)
        self.assertAlmostEqual(v['drift']['candidate_actual_speedup'],1)
        self.assertAlmostEqual(v['drift']['reference_time_ratio'],2)
    def test_largest_input_routes_to_device(self):
        p=self.result('candidate',[1,1],[1,4]);v=m.analyze(p,self.source,self.catalog)
        self.assertEqual(v['cases'][0]['case'],'example-timing-1')
        self.assertEqual(v['hypotheses'][0]['direction'],'large_input_device_path')
        self.assertAlmostEqual(v['score'],.5)
    def test_reject_stale_source(self):
        p=self.result('candidate',[1,1],[1,1]);self.source.write_text('changed')
        with self.assertRaisesRegex(ValueError,'身份'):m.analyze(p,self.source,self.catalog)
    def test_reject_other_contract(self):
        old=self.result('old',[1,1],[2,2]);new=self.result('new',[1,1],[1,1])
        p=old.with_name('old.request.json');v=json.loads(p.read_text());v['binding']['bundle_id']='different';p.write_text(json.dumps(v))
        with self.assertRaisesRegex(ValueError,'合同绑定'):m.analyze(new,self.source,self.catalog,parent=old)
    def test_reject_invalid_measurement(self):
        p=self.result('candidate',[1,1],[1,0])
        with self.assertRaisesRegex(ValueError,'无效耗时'):m.analyze(p,self.source,self.catalog)

if __name__=='__main__':unittest.main()
