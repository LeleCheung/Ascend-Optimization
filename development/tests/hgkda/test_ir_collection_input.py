import json
from pathlib import Path
import tempfile
import unittest

from hgkda.runtime.ir_input import freeze, install, sha


class IRCollectionInputTests(unittest.TestCase):
    def prepare(self, root):
        bundle = root / 'bundle'
        (bundle / 'ir').mkdir(parents=True)
        (bundle / 'frozen.py').write_text('original operator source', encoding='utf8')
        (bundle / 'ir/kernel.ttir').write_text('changed IR', encoding='utf8')
        (bundle / 'ir_override_runtime.py').write_text('raise RuntimeError("compiler reached")', encoding='utf8')
        manifest = dict(source='frozen.py', source_sha256=sha(bundle / 'frozen.py'),
            overrides=[dict(path='ir/kernel.ttir', sha256=sha(bundle / 'ir/kernel.ttir'))],
            cache_root='old-cache', receipt_directory='old-receipts')
        path = bundle / 'ir-manifest.json'
        path.write_text(json.dumps(manifest), encoding='utf8')
        return path

    def test_collection_preserves_source_and_ir_and_rejects_changed_resource(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.prepare(root)
            source, binding = freeze(path, root / 'capture', path.parent / 'frozen.py')
            self.assertEqual(binding['source_sha256'], sha(path.parent / 'frozen.py'))
            self.assertEqual(sha(source), binding['source_sha256'])
            self.assertEqual(binding['original_manifest_sha256'], sha(path))
            self.assertEqual((source.parent / 'ir/kernel.ttir').read_bytes(), (path.parent / 'ir/kernel.ttir').read_bytes())
            (source.parent / 'ir/kernel.ttir').write_text('tampered IR', encoding='utf8')
            with self.assertRaisesRegex(AssertionError, 'resource changed'):
                install(dict(source=str(source), source_sha256=sha(source), ir_input=binding))

    def test_resource_traversal_is_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.prepare(root)
            manifest = json.loads(path.read_bytes())
            manifest['overrides'][0]['path'] = '../outside.ttir'
            path.write_text(json.dumps(manifest), encoding='utf8')
            with self.assertRaisesRegex(AssertionError, 'escapes'):
                freeze(path, root / 'capture', path.parent / 'frozen.py')
            self.assertFalse((root / 'capture').exists())

    def test_analysis_runtime_transport_change_preserves_computational_resources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.prepare(root)
            transport = root / 'new_runtime.py'
            transport.write_text('updated generic transport', encoding='utf8')
            source, binding = freeze(path, root / 'capture', path.parent / 'frozen.py', runtime_path=transport)
            self.assertEqual(binding['runtime_sha256'], sha(transport))
            self.assertNotEqual(binding['runtime_sha256'], binding['original_runtime_sha256'])
            self.assertEqual(sha(source), sha(path.parent / 'frozen.py'))
            self.assertEqual(sha(source.parent / 'ir/kernel.ttir'), sha(path.parent / 'ir/kernel.ttir'))


if __name__ == '__main__':
    unittest.main()
