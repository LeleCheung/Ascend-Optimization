"""Check that replay evidence cannot silently change IR resources or binaries."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from development.tools.research.ir_observation_binding import observed


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ObservationBindingTest(unittest.TestCase):
    def fixture(self, root):
        source = root / 'frozen.py'
        source.write_bytes(b'fixed source')
        ir = root / 'kernel.ttir'
        ir.write_bytes(b'fixed model-authored IR')
        binary = root / 'kernel.npubin'
        binary.write_bytes(b'formal kernel binary')
        override = dict(ast_sha256='target', path=ir.name, sha256=sha(ir))
        manifest = dict(source_sha256=sha(source), overrides=[override], receipt_directory=str(root / 'receipts'))
        path = root / 'ir-manifest.json'
        path.write_text(json.dumps(manifest), encoding='utf8')
        receipt = root / 'receipts/one.jsonl'
        receipt.parent.mkdir()
        receipt.write_text(json.dumps(dict(source_sha256=sha(source), manifest_sha256=sha(path),
            override_applied=True, ast_sha256='target', ir_sha256=sha(ir),
            artifacts=[dict(kind='npubin', path=str(binary), sha256=sha(binary))])) + '\n', encoding='utf8')
        spec = dict(source=str(source), source_sha256=sha(source), ir_input=dict(manifest=str(path),
            manifest_sha256=sha(path), source_sha256=sha(source),
            resources={source.name: sha(source), ir.name: sha(ir)}))
        return spec, manifest, dict(target=sha(binary))

    def test_matching_resources_and_binary_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec, manifest, formal = self.fixture(Path(temporary))
            self.assertEqual(observed(spec, manifest, formal, {'target'})['binaries'], formal)

    def test_mutable_ir_and_different_formal_binary_are_rejected(self):
        for kind in ['ir', 'binary']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                spec, manifest, formal = self.fixture(root)
                if kind == 'ir':
                    (root / 'kernel.ttir').write_bytes(b'changed after capture')
                else:
                    formal['target'] = 'different formal binary'
                with self.assertRaisesRegex(AssertionError, 'resource bytes changed' if kind == 'ir' else 'differs from formal'):
                    observed(spec, manifest, formal, {'target'})


if __name__ == '__main__':
    unittest.main()
