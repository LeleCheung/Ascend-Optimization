"""控制器结果选择遵循实际耗时，且准备器保持原合同字节。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

TOOLS=Path(__file__).resolve().parents[1]


def load(name,filename):
    s=importlib.util.spec_from_file_location(name,TOOLS/filename)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


class EnduranceTests(unittest.TestCase):
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
