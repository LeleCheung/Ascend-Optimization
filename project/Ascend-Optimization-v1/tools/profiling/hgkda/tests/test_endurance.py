"""控制器结果选择遵循实际耗时，且准备器保持原合同字节。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import hashlib
import io
import tarfile

TOOLS=Path(__file__).resolve().parents[1]


def load(name,filename):
    s=importlib.util.spec_from_file_location(name,TOOLS/filename)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


class EnduranceTests(unittest.TestCase):
    def test_scan_timing_gate_uses_frozen_large_inputs(self):
        m=load('endurance_semantics','check-endurance.py')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);leaf=root/'catalog/ops/cumulative_trapezoid';leaf.mkdir(parents=True)
            files={'oracle.py':b'def run(y): return y\n',
                   'timing.jsonl':b'{"name":"large-dot","inputs":{"spec":{"shape":[1024,4096]}}}\n'}
            for name,data in files.items():(leaf/name).write_bytes(data)
            record=dict(operator='cumulative_trapezoid',catalog_leaf='catalog/ops/cumulative_trapezoid',timing_cases=1,
                contract_sha256={n:hashlib.sha256(d).hexdigest() for n,d in files.items()})
            (root/'plan.json').write_text(json.dumps(dict(records=[record])))
            source=root/'results/candidate/main.py';source.parent.mkdir(parents=True);source.write_text('')
            observed,rows=m.cumulative_timing_contract(source)
            self.assertEqual(observed,leaf);self.assertEqual(rows[0]['inputs']['spec']['shape'],[1024,4096])
            (leaf/'timing.jsonl').write_text('')
            with self.assertRaises(ValueError):m.cumulative_timing_contract(source)

    def test_uploaded_contract_and_individual_cases_are_verified(self):
        m=load('endurance_audit','audit-endurance.py')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            contents={'definition.json':b'{}','oracle.py':b'def run(A): return A\n',
                'correctness.jsonl':b'{"name":"check-1"}\n','timing.jsonl':b'{"name":"time-1"}\n'}
            with tarfile.open(root/'operator-bundle.tar','w') as archive:
                for name,data in contents.items():
                    item=tarfile.TarInfo(name);item.size=len(data);archive.addfile(item,io.BytesIO(data))
            record=dict(contract_sha256={k:hashlib.sha256(v).hexdigest() for k,v in contents.items()})
            (root/'inspect.json').write_text(json.dumps(dict(benchmark_fingerprint='fixed-contract')))
            (root/'once.request.json').write_text(json.dumps(dict(binding=dict(bundle_id='sha256:'+m.sha(root/'operator-bundle.tar')))))
            result=dict(benchmark_fingerprint='fixed-contract',per_workload=[dict(uuid='check-1',phase='correctness',status='PASSED'),
                dict(uuid='time-1',phase='timing',status='PASSED')])
            m.verify_formal_contract(record,root,result)
            result['per_workload'][1]['uuid']='different-case'
            with self.assertRaises(ValueError):m.verify_formal_contract(record,root,result)
            result['per_workload'][1]['uuid']='time-1'
            record['contract_sha256']['oracle.py']='0'*64
            with self.assertRaises(ValueError):m.verify_formal_contract(record,root,result)

    def test_actual_latency_ranking_ignores_reference_drift(self):
        m=load('endurance_logic','run-endurance.py')
        parent=dict(geo_mean=.75,per_workload=[dict(uuid='a',phase='timing',latency_ms=.020),dict(uuid='b',phase='timing',latency_ms=.080)])
        faster=dict(geo_mean=.70,per_workload=[dict(uuid='b',phase='timing',latency_ms=.040),dict(uuid='a',phase='timing',latency_ms=.010)])
        self.assertAlmostEqual(m.improved(parent,faster),2.)
        self.assertLess(m.latency(faster),m.latency(parent))
        faster['per_workload'][0]['uuid']='foreign'
        with self.assertRaises(ValueError):m.improved(parent,faster)

    def test_prepare_keeps_contract_and_latest_selu(self):
        m=load('endurance_prepare','prepare-endurance.py')
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'new-batch';plan=m.prepare(out)
            self.assertEqual([r['operator'] for r in plan['records']],m.ORDER)
            self.assertEqual(plan['final_repeats'],1)
            previous=m.PROJECT/'operators/campaigns/night-20261021-v3'
            for row in plan['records']:
                leaf=Path(row['catalog_leaf'])
                for filename in row['contract_sha256']:
                    self.assertEqual((out/leaf/filename).read_bytes(),(previous/leaf/filename).read_bytes())
            self.assertEqual(plan['records'][0]['source_sha256'],'49bdbb789d497b6423f893f3433167f3fd4bc61197676236bd62a9ec79c7dd91')
            with self.assertRaises(FileExistsError):m.prepare(out)


if __name__=='__main__':unittest.main()
