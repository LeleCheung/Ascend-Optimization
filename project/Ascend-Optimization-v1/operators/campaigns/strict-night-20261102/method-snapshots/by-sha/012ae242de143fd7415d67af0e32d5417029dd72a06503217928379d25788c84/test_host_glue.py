"""宿主桥的人工豁免不能覆盖Torch数值计算或被替换的二进制。"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from host_glue import admit


class HostGlueTests(unittest.TestCase):
    def fixture(self, folder, numerical=False):
        root = Path(folder)
        cpp = root/'host.cpp'
        binary = root/'host.so'
        cpp.write_text('// reviewed metadata bridge')
        binary.write_bytes(b'fixture binary identity')
        source = root/'main.py'
        source.write_text('import torch\nimport importlib.util\nfrom pathlib import Path\n'
                          +'binary = '+repr(str(binary))+'\n'
                          +'def run(x):\n    return '+('torch.add(x, x)' if numerical else 'x')+'\n')
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        review = dict(source_sha256=sha(source), status='reviewed_strict_triton',
                      all_branches_reviewed=True, dependencies_reviewed=True,
                      reviewed_host_bridges=[dict(candidate_sha256=sha(source), status='reviewed_host_only',
                          all_functions_reviewed=True, device_computation='Triton_only', functions=['execute'],
                          binary=str(binary), binary_sha256=sha(binary),
                          cpp_source=str(cpp), cpp_source_sha256=sha(cpp))])
        path = root/'review.json'
        path.write_text(json.dumps(review))
        return source, path, binary

    def test_bound_bridge_passes_and_replaced_binary_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            source, review, binary = self.fixture(folder)
            self.assertEqual(admit(source, review)['status'], 'reviewed_host_only')
            binary.write_bytes(b'replaced')
            with self.assertRaises(AssertionError):
                admit(source, review)

    def test_review_does_not_exempt_torch_numerical_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            source, review, _ = self.fixture(folder, numerical=True)
            with self.assertRaises(AssertionError):
                admit(source, review)


if __name__ == '__main__':
    unittest.main()
