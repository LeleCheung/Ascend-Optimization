"""使用真实 master 证据验证对比工具的门禁，避免把失败/异口径结果列为成果。"""
import copy
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('compare_versions', PROJECT / 'tools/evaluation/compare-operator-versions.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
REPORT = PROJECT / 'operators/matmul_bias_activation/reports/master-20261010'


class CompareTests(unittest.TestCase):
    def test_real_master_identity(self):
        master = MODULE.load_result(REPORT / 'flaggems-master-1.result.json')
        summary = MODULE.compare([('master', master)])
        self.assertEqual(summary[0]['correctness_count'], 27)
        self.assertEqual(summary[0]['timing_count'], 15)
        self.assertAlmostEqual(summary[0]['relative_master'], 1.0)

    def test_different_fingerprint_and_workloads_rejected(self):
        master = MODULE.load_result(REPORT / 'flaggems-master-1.result.json')
        for field in ('fingerprint', 'rows', 'timing_scope'):
            different = copy.deepcopy(master)
            if field == 'timing_scope':
                different[field] = 'walltime'
            elif field == 'fingerprint':
                different[field] = 'tampered'
            else:
                different[field].pop(next(iter(different[field])))
            with self.assertRaises(ValueError):
                MODULE.compare([('master', master), ('changed', different)])

    def test_source_snapshot_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for name in ('flaggems-master-1.result.json', 'flaggems-master-1.request.json',
                         'flaggems-master.provenance.json', 'flaggems-master.py', 'inspect.json'):
                shutil.copyfile(REPORT / name, folder / name)
            with (folder / 'flaggems-master.py').open('a', encoding='utf-8') as stream:
                stream.write('\n# changed\n')
            with self.assertRaises(ValueError):
                MODULE.load_result(folder / 'flaggems-master-1.result.json')

    def test_failed_candidate_rejected(self):
        failed = PROJECT / 'operators/matmul_bias_activation/reports/profiling-k128-pipeline-20261010/profiling-k128-pipeline-1.result.json'
        with self.assertRaises(ValueError):
            MODULE.load_result(failed)


if __name__ == '__main__':
    unittest.main()
