"""验证后端门禁能区分Triton内存操作与宿主厂商／Tensor替代路径。"""
import tempfile
import unittest
from pathlib import Path
from common import save,sha
from gate import inspect,admit

class BackendGate(unittest.TestCase):
    def inspect_text(self,text):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'candidate.py';p.write_text(text,encoding='utf8')
            return inspect(p)
    def test_compiler_dma_is_allowed(self):
        r=self.inspect_text('import triton\nimport triton.language as tl\n@triton.jit\ndef k(X,Y):\n    tl.store(Y,tl.load(X))\n')
        self.assertTrue(r['static_passed']);self.assertEqual(r['triton_kernels'],['k'])
    def test_host_copy_and_fill_are_rejected(self):
        for text in ['def run(x):\n return x.clone()','def run(x):\n return x.contiguous()',
                     'def run(x,y):\n return y.copy_(x)','import torch\ndef run(x):\n return torch.zeros(x.shape)']:
            with self.subTest(text=text):self.assertFalse(self.inspect_text(text)['static_passed'])
    def test_external_native_library_is_rejected(self):
        self.assertFalse(self.inspect_text('p="/tmp/libkernel.so"')['static_passed'])
        self.assertFalse(self.inspect_text('import ctypes\ndef run(x):\n return ctypes.CDLL("libascendcl.so")')['static_passed'])
    def test_torch_target_and_dynamic_execution_rejected(self):
        self.assertFalse(self.inspect_text('import torch\ndef run(x):\n return torch.ops.aten.neg(x)')['static_passed'])
        self.assertFalse(self.inspect_text('def run(x):\n return eval(x)')['static_passed'])
    def test_manual_review_bound_to_exact_source(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'candidate.py';p.write_text('def run(x):\n return x\n',encoding='utf8')
            review=Path(folder)/'review.json'
            save(review,dict(status='reviewed_strict_triton',source_sha256=sha(p),all_branches_reviewed=True,dependencies_reviewed=True))
            admit(p,review)
            p.write_text('def run(x):\n return x\n# changed\n',encoding='utf8')
            with self.assertRaises(AssertionError):admit(p,review)
    def test_reviewed_host_bridge_requires_exact_binary_and_cpp(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);binary=folder/'host.so';binary.write_bytes(b'host test identity')
            cpp=folder/'host.cpp';cpp.write_bytes(b'// reviewed host glue')
            p=folder/'candidate.py';p.write_text('p='+repr(binary.as_posix())+'\n',encoding='utf8')
            bridge=dict(binary=str(binary),binary_sha256=sha(binary),cpp_source=str(cpp),cpp_source_sha256=sha(cpp),
                candidate_sha256=sha(p),status='reviewed_host_only',all_functions_reviewed=True,device_computation='Triton_only')
            review=folder/'review.json';save(review,dict(status='reviewed_strict_triton',source_sha256=sha(p),
                all_branches_reviewed=True,dependencies_reviewed=True,reviewed_host_bridges=[bridge]))
            self.assertFalse(inspect(p)['static_passed'])
            self.assertTrue(admit(p,review)['admitted'])
            binary.write_bytes(b'tampered')
            with self.assertRaises(AssertionError):admit(p,review)

if __name__=='__main__':unittest.main()
