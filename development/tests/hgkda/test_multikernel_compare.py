"""多kernel参考保留调用次数与原始分布，不把任务时间相加冒充完整耗时。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('multikernel_compare',ROOT/'hgkda/src/hgkda/compare.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class MultiKernelTests(unittest.TestCase):
    def test_timestamp_suffixed_profiler_tables_keep_original_hashes(self):
        original=ROOT/'hgkda/examples/paired-expm1/paired/reference'
        meta=json.loads((original/'collection.json').read_bytes());digest=meta['spec_sha256']
        expected=module.read_side(original,{digest},'reference')
        with tempfile.TemporaryDirectory() as temporary:
            copied=Path(temporary)/'reference';shutil.copytree(original,copied)
            for table in (copied/'profile').rglob('*.csv'):
                table.rename(table.with_name(table.stem+'_20261116202314061.csv'))
            result=module.read_side(copied,{digest},'reference')
            self.assertEqual(result['tables'],expected['tables'])
            self.assertTrue(all('_20261116202314061.csv' in v['path'] for v in result['evidence'].values()))
    def test_occurrences_weight_counts_and_keep_latency_unknown(self):
        original=ROOT/'hgkda/examples/paired-expm1/paired/reference'
        meta=json.loads((original/'collection.json').read_bytes());digest=meta['spec_sha256']
        single=module.read_side(original,{digest},'reference')
        with tempfile.TemporaryDirectory() as temporary:
            parent=Path(temporary)
            for name in ['first','second']:shutil.copytree(original,parent/name)
            (parent/'collection.json').write_text(json.dumps(dict(status='completed',spec_sha256=digest,side='reference',groups=meta['groups'],
                kernels=[dict(directory='first',symbol=meta['expected_kernel'],occurrences=1),dict(directory='second',symbol=meta['expected_kernel'],occurrences=2)])),encoding='utf8')
            multi=module.read_side(parent,{digest},'reference')
            self.assertIsNone(multi['basic']['Task Duration(us)'])
            self.assertEqual(len(multi['kernel_profiles']),2)
            for key,table in single['tables'].items():
                self.assertEqual(multi['tables'][key]['rows'],table['rows']*3)
                for field,value in table['fields'].items():
                    actual=multi['tables'][key]['fields'][field]
                    if value['sum'] is not None:self.assertAlmostEqual(actual['sum'],value['sum']*3,places=7)
                    self.assertEqual(actual['missing'],value['missing']*3)
            for key,value in single['derived'].items():
                if value is not None:self.assertAlmostEqual(multi['derived'][key],value)


if __name__=='__main__':unittest.main()
