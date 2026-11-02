"""验证准入边界：依赖/别名不能绕过，Triton设备转换必须允许。"""
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('night_admission',Path(__file__).with_name('admission.py'))
admission=importlib.util.module_from_spec(spec);spec.loader.exec_module(admission)


class BoundaryTests(unittest.TestCase):
    def test_tensor_computation_and_external_execution_rejected(self):
        for source in [
            'import torch as t\ndef run(x): return t.sin(x)',
            'from torch import neg as f\ndef run(x): return f(x)',
            'def run(x): return x.copy_(x)',
            'import ctypes', 'import acl', 'import flag_gems',
            "def run(x): return __import__('acl')",
        ]:
            with self.subTest(source=source):self.assertTrue(admission.static(source))

    def test_device_computation_and_host_metadata_permitted(self):
        source='''import torch
import triton
import triton.language as tl
@triton.jit
def kernel(X,Y,N:tl.constexpr):
    v=tl.load(X).to(tl.float32)
    tl.store(Y,v)
def run(x):
    shape=torch.broadcast_shapes(x.shape,x.shape)
    y=torch.empty(shape,device=x.device,dtype=x.dtype)
    kernel[(1,)](x,y,x.numel())
    return y
'''
        self.assertEqual(admission.static(source),[])

    def test_torch_inside_jit_rejected(self):
        self.assertTrue(admission.static('''import torch
import triton
@triton.jit
def kernel(x):
    return torch.sin(x)
'''))

    def test_exact_evaluation_reused_unknown_not_retried(self):
        class Model:
            def __init__(self,**value):self.value=value
            def model_dump(self,**kwargs):return self.value
        bundle=SimpleNamespace(kernel_code='first',binding=Model(bundle_id='frozen'),
                               solution=Model(name='round1',sources=['first'],entrypoint='main.py::run'))
        context=SimpleNamespace(target_hardware='910B')
        called=[]
        def evaluate(*args,**kwargs):
            called.append(True);return dict(status='PASSED',geo_mean=.9),dict(backend='npu')
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,HGKDA_STRICT_BATCH=folder):
            a=admission.cached_evaluate(evaluate,bundle,context,dict(warmup_ms=1000))
            bundle.solution.value['name']='round2'
            b=admission.cached_evaluate(evaluate,bundle,context,dict(warmup_ms=1000))
            self.assertEqual(a,b);self.assertEqual(len(called),1)
            bundle.kernel_code='second'
            def interrupted(*args,**kwargs):raise RuntimeError('lost transport')
            with self.assertRaisesRegex(RuntimeError,'lost transport'):
                admission.cached_evaluate(interrupted,bundle,context,{})
            with self.assertRaisesRegex(RuntimeError,'未知'):
                admission.cached_evaluate(evaluate,bundle,context,{})
            self.assertEqual(len(called),1)


if __name__=='__main__':unittest.main()
