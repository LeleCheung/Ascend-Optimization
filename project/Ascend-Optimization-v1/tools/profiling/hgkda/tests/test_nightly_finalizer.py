"""最终核验不得因重复运行或检查器改变而覆盖原始失败。"""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

spec=importlib.util.spec_from_file_location('night_finalizer',Path(__file__).resolve().parents[1]/'finalize-nightly.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class FinalizerTests(unittest.TestCase):
    def test_failed_same_identity_is_reused_without_writing(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'semantics.json'
            data={'status':'failed','source_sha256':'source','checker_sha256':'checker','passed':47,'total':48}
            before=json.dumps(data).encode();path.write_bytes(before)
            self.assertEqual(m.existing_semantics(path,'source','checker'),data)
            self.assertEqual(path.read_bytes(),before)

    def test_changed_checker_requires_separate_attempt(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'semantics.json'
            before=b'{"status":"failed","source_sha256":"source","checker_sha256":"old"}'
            path.write_bytes(before)
            with self.assertRaisesRegex(RuntimeError,'身份不同'):m.existing_semantics(path,'source','new')
            self.assertEqual(path.read_bytes(),before)

    def test_missing_file_allows_first_check(self):
        with TemporaryDirectory() as directory:
            self.assertIsNone(m.existing_semantics(Path(directory)/'missing.json','source','checker'))


if __name__=='__main__':unittest.main()
