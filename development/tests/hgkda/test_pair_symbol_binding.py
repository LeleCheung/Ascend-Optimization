import unittest

from hgkda.runtime.collect_pair import resolve_collected_symbol, partition_profile_tasks


class PairSymbolBindingTests(unittest.TestCase):
    def test_mixed_core_tasks_require_explicit_known_type(self):
        vector={'name':'scatter','task_type':'KERNEL_MIX_AIV'}
        cube={'name':'matmul','task_type':'KERNEL_MIX_AIC'}
        self.assertEqual(partition_profile_tasks([vector,cube]),([vector,cube],[]))
        with self.assertRaises(ValueError):
            partition_profile_tasks([{'task_type':'KERNEL_MIX_UNKNOWN'}])

    def test_dma_preserved_separately_from_core_metrics(self):
        compute={'name':'bn','task_type':'KERNEL_AIVEC'}
        dma={'name':'MEMCPY_ASYNC','task_type':'SDMA_SQE'}
        self.assertEqual(partition_profile_tasks([compute,dma,dma]),([compute],[dma,dma]))
        with self.assertRaises(ValueError):partition_profile_tasks([{'task_type':'UNKNOWN'}])
        with self.assertRaises(ValueError):partition_profile_tasks([dma])

    def test_unique_reported_reference_specialization(self):
        actual='TransformBiasRescaleQkv_float16_high_performance_1'
        log=f'Start analyze kernel on device: device7, kernel: {actual}, name is: 0'
        self.assertEqual(resolve_collected_symbol('TransformBiasRescaleQkv',actual,log,'reference'),actual)

    def test_requires_matching_unique_reference_evidence(self):
        actual='Op_float16_1';line=f'Start analyze kernel on device: device7, kernel: {actual}, name is: 0'
        for alias,log,side in [('Other',line,'reference'),('Op',line,'candidate'),('Op','', 'reference'),
            ('Op',line+'\nStart analyze kernel on device: device7, kernel: Op_other, name is: 1','reference')]:
            with self.subTest(alias=alias,side=side,log=log),self.assertRaises(ValueError):
                resolve_collected_symbol(alias,actual,log,side)
