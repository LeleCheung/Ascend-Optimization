"""OpProf失败请求中的有效设备计数必须保留范围、缺失值及原始身份。"""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('opprof', Path(__file__).resolve().parents[1]/'opprof_facts.py')
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class OpProfTests(unittest.TestCase):
    def fixture(self, root, *, extra=None, missing=False):
        files = {
            'OpBasicInfo.csv': 'Op Name,Task Duration(us),Block Dim,\nreduce,20,2,\n',
            'Memory.csv': 'block_id,sub_block_id,read_main_memory_datas(KB),\n0,vector0,256,\n1,vector0,'+('NA' if missing else '512')+',\n',
            'PipeUtilization.csv': 'block_id,sub_block_id,aiv_vec_ratio,aiv_mte2_ratio,\n0,vector0,.8,.7,\n1,vector0,.6,.9,\n',
            'L2Cache.csv': 'block_id,sub_block_id,aiv_r0_read_cache_hit,aiv_r0_read_cache_miss_allocate,aiv_r1_read_cache_hit,aiv_r1_read_cache_miss_allocate,\n0,vector0,1,1,0,0,\n1,vector0,1,97,0,0,\n',
        }
        path = root/'report.tar.gz'
        with tarfile.open(path, 'w:gz') as archive:
            for name, content in files.items():
                raw=content.encode();member=tarfile.TarInfo('./kernel-000/report/'+name)
                member.size=len(raw);archive.addfile(member,io.BytesIO(raw))
            if extra:
                member=tarfile.TarInfo(extra);member.size=1;archive.addfile(member,io.BytesIO(b'x'))
        manifest={'request': {'identity': {'solution_sha256':'solution'}, 'case_id':'case'},
                  'result': {'profile_id':'profile', 'status':'failed', 'error':'simulator timeout',
                             'artifacts':[{'kind':'vendor_kernel_report','filename':'msprof-op-report.tar.gz',
                                           'size_bytes':path.stat().st_size}]}}
        args={'expected_archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'expected_solution_sha256':'solution','expected_case_id':'case','expected_kernel':'reduce'}
        return path,manifest,args

    def test_partial_device_facts_do_not_claim_instruction_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path,manifest,args=self.fixture(Path(directory));data=m.decode(path,manifest,**args)
            self.assertTrue(data['partial_collection']);self.assertFalse(data['instruction_listing_available'])
            self.assertEqual(data['l2_read_hit_rate_pct']['aiv']['weighted_hit_rate_pct'],2.)
            self.assertEqual(data['memory_reported_totals_KB']['read_main_memory_datas(KB)'],768.)
            self.assertEqual(data['tables']['PipeUtilization.csv']['numeric']['aiv_vec_ratio']['mean'],.7)
            self.assertNotIn('serialization_pct',data)

    def test_unknown_counters_do_not_become_zero_or_partial_total(self):
        with tempfile.TemporaryDirectory() as directory:
            path,manifest,args=self.fixture(Path(directory),missing=True);data=m.decode(path,manifest,**args)
            self.assertIsNone(data['memory_reported_totals_KB']['read_main_memory_datas(KB)'])
            self.assertNotIn('aic',data['l2_read_hit_rate_pct'])

    def test_archive_case_candidate_and_kernel_identity_are_required(self):
        for key in ['expected_archive_sha256','expected_solution_sha256','expected_case_id','expected_kernel']:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as directory:
                path,manifest,args=self.fixture(Path(directory));args[key]='foreign'
                with self.assertRaises(ValueError):m.decode(path,manifest,**args)

    def test_traversal_and_duplicate_tables_are_rejected_without_extracting(self):
        for extra in ['../escape.txt','kernel-000/other/Memory.csv']:
            with self.subTest(extra=extra),tempfile.TemporaryDirectory() as directory:
                path,manifest,args=self.fixture(Path(directory),extra=extra)
                with self.assertRaises(ValueError):m.decode(path,manifest,**args)


if __name__ == '__main__':unittest.main()
