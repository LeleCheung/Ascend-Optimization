"""字节码不应阻断KG，合同变化和其他未知文件不能被静默隐藏。"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from catalog_snapshot import freeze_catalog


class CatalogSnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.source=self.root/'catalog';self.leaf=self.source/'ops/x'
        self.leaf.mkdir(parents=True);(self.leaf/'oracle.py').write_bytes(b'def run(x): return x\n')
        self.record=dict(catalog_leaf='catalog/ops/x',contract_sha256={
            'oracle.py':hashlib.sha256((self.leaf/'oracle.py').read_bytes()).hexdigest()})

    def test_generated_bytecode_excluded_original_preserved_other_files_kept(self):
        cache=self.leaf/'__pycache__';cache.mkdir();(cache/'oracle.cpython-311.pyc').write_bytes(b'cache')
        (self.leaf/'unexpected.txt').write_bytes(b'KG must see this file')
        snapshot=freeze_catalog(self.source,self.root/'snapshot',[self.record])
        self.assertFalse((snapshot/'ops/x/__pycache__').exists())
        self.assertTrue((cache/'oracle.cpython-311.pyc').exists())
        self.assertEqual((snapshot/'ops/x/unexpected.txt').read_bytes(),b'KG must see this file')
        self.assertEqual((snapshot/'ops/x/oracle.py').read_bytes(),(self.leaf/'oracle.py').read_bytes())

    def test_contract_change_rejected_before_copy(self):
        (self.leaf/'oracle.py').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA|冻结合同'):
            freeze_catalog(self.source,self.root/'snapshot',[self.record])
        self.assertFalse((self.root/'snapshot').exists())

    def test_existing_snapshot_not_overwritten(self):
        snapshot=freeze_catalog(self.source,self.root/'snapshot',[self.record])
        (snapshot/'ops/x/oracle.py').write_bytes(b'changed snapshot')
        with self.assertRaises(ValueError):freeze_catalog(self.source,snapshot,[self.record])
        self.assertEqual((snapshot/'ops/x/oracle.py').read_bytes(),b'changed snapshot')

    def test_nonbytecode_in_cache_not_silently_hidden(self):
        cache=self.leaf/'__pycache__';cache.mkdir();(cache/'notes.md').write_bytes(b'not generated')
        with self.assertRaises(ValueError):freeze_catalog(self.source,self.root/'snapshot',[self.record])


if __name__=='__main__':unittest.main()
