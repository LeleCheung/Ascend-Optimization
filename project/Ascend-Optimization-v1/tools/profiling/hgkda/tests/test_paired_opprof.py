"""成对采集不能混用输入身份，且必须保留core分类与零分母。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import hashlib

path = Path(__file__).resolve().parents[1]/'skills/hgkda/scripts/compare_opprof.py'
spec = importlib.util.spec_from_file_location('paired_compare', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PairedTest(unittest.TestCase):
    def fixture(self, root):
        (root/'spec.json').write_text('{"operator":"log10_"}')
        digest = hashlib.sha256((root/'spec.json').read_bytes()).hexdigest()
        for side, name, count in [('reference', 'Log_ref', 4), ('candidate', 'log10_ker', 16)]:
            directory = root/side
            directory.mkdir()
            (directory/'collection.json').write_text(json.dumps(dict(status='completed',
                spec_sha256=digest, side=side, expected_kernel=name, groups='Roofline')))
            prof = directory/'profile'
            prof.mkdir()
            (prof/'OpBasicInfo.csv').write_text('Op Name,Device Id,Task Duration(us)\n'+name+',0,10\n')
            (prof/'PipeUtilization.csv').write_text('block_id,sub_block_id,aiv_vec_ratio\n0,vector0,.7\n1,vector0,.9\n0,cube0,NA\n')
            (prof/'Memory.csv').write_text('block_id,sub_block_id,GM_to_UB_datas(KB),aiv_mte2_instructions\n'
                '0,vector0,1,'+str(count)+'\n1,vector0,1,'+str(count)+'\n0,cube0,NA,NA\n')

    def test_bytes_per_instruction_and_core_scope(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p);result=m.compare(p)
            derived=result['derived_differences']['Memory.csv/vector0/bytes_per_mte2']
            self.assertEqual(derived['reference'],256)
            self.assertEqual(derived['candidate'],64)
            self.assertEqual(derived['candidate_over_reference'],.25)
            self.assertIsNone(result['candidate']['derived']['Memory.csv/cube0/bytes_per_mte2'])
            self.assertEqual(result['candidate']['tables']['PipeUtilization.csv/vector0']['fields']['aiv_vec_ratio']['mean'],.8)

    def test_foreign_spec_or_target_rejected(self):
        for key in ['spec_sha256','expected_kernel']:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as d:
                p=Path(d);self.fixture(p);path=p/'candidate/collection.json'
                meta=json.loads(path.read_text());meta[key]='foreign';path.write_text(json.dumps(meta))
                with self.assertRaises(ValueError):m.compare(p)


if __name__ == '__main__':
    unittest.main()
