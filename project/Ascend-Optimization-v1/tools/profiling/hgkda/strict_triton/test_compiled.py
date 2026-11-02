"""编译证明门禁：空清单、缺少设备产物或身份变化均不能算后端核验成功。"""
import tempfile
import unittest
from pathlib import Path
from common import save,read,sha
from verify_compiled import validate_manifest


class CompiledProof(unittest.TestCase):
    def fixture(self,folder):
        folder=Path(folder);source=folder/'main.py';source.write_bytes(b'# frozen test candidate')
        artifacts=[]
        for kind in ['ttir','npubin']:
            path=folder/('kernel.'+kind);path.write_bytes(('test-'+kind).encode())
            artifacts.append(dict(kind=kind,path=str(path),sha256=sha(path)))
        manifest=folder/'manifest.json';save(manifest,dict(spec=dict(source_sha256=sha(source)),
            entries=[dict(symbol='kernel',metadata='test fixture',artifacts=artifacts)]))
        return source,manifest

    def test_empty_manifest_and_missing_compiler_stage_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source,manifest=self.fixture(folder);original=read(manifest)
            self.assertEqual(validate_manifest(manifest,source)['status'],'verified')
            save(manifest,dict(original,entries=[]))
            with self.assertRaises(AssertionError):validate_manifest(manifest,source)
            for missing in ['ttir','npubin']:
                value=read(manifest);value['entries']=[dict(original['entries'][0],artifacts=[a for a in original['entries'][0]['artifacts'] if a['kind']!=missing])]
                save(manifest,value)
                with self.assertRaises(AssertionError):validate_manifest(manifest,source)

    def test_tampered_device_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source,manifest=self.fixture(folder)
            (Path(folder)/'kernel.npubin').write_bytes(b'changed')
            with self.assertRaises(AssertionError):validate_manifest(manifest,source)

    def test_foreign_source_identity_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source,manifest=self.fixture(folder);source.write_bytes(b'foreign candidate')
            with self.assertRaises(AssertionError):validate_manifest(manifest,source)


if __name__=='__main__':unittest.main()
