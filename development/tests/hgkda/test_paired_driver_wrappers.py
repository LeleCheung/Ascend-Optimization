import unittest
from types import SimpleNamespace
from hgkda.runtime.paired_driver import wrapped_kernels, clear_launch_caches


class PairedDriverWrapperTests(unittest.TestCase):
    def test_nested_tuner_keeps_configuration_cache(self):
        tuner=SimpleNamespace(configs=[64],cache={(8,):64},fn=None)
        wrapper=SimpleNamespace(fn=tuner,kernel_cache=({'compiled':'binary'},{}))
        module=SimpleNamespace(kernel=wrapper)
        self.assertEqual(list(wrapped_kernels(module)),[('kernel',1,tuner)])
        clear_launch_caches(module)
        self.assertEqual(wrapper.kernel_cache,({},{}))
        self.assertEqual(tuner.cache,{(8,):64})

    def test_direct_tuner_and_cycle_terminate(self):
        tuner=SimpleNamespace(configs=[64],cache={},fn=None)
        tuner.fn=tuner
        self.assertEqual(list(wrapped_kernels(SimpleNamespace(kernel=tuner))),[('kernel',0,tuner)])
