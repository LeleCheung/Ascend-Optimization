"""验证机制推断边界：拒绝跨范围推断，不把热点或噪声当根因。"""
import copy
import unittest
from core import diagnose


class MechanismTests(unittest.TestCase):
    def probe(self):
        return dict(schema='hgkda-execution-slice-v1',source_sha256='a'*64,cases=[dict(id='case',status='passed',
            timings=dict(wrapper_us=[90,91,89],direct_us=[40,41,39]),launches=[dict(symbol='kernel',programs=1)])])

    def test_dispatch_intervention_supports_dispatch_not_device_edges(self):
        result=diagnose(self.probe())
        self.assertEqual(result['decisions'][0]['action'],'cache_compiled_dispatch')
        self.assertEqual(len(result['graphs'][0]['edges']),4)
        self.assertTrue(any('设备内部' in s for s in result['graphs'][0]['unknown']))

    def test_noisy_direct_intervention_does_not_confirm_dispatch(self):
        p=self.probe();p['cases'][0]['timings']['direct_us']=[50,90,100]
        self.assertFalse(diagnose(p)['decisions'][0]['competing_hypotheses'][0]['supported'])

    def test_failed_probe_retained(self):
        p=self.probe();p['cases'][0].update(status='failed',error='unsupported')
        result=diagnose(p);self.assertEqual(result['graphs'],[])
        self.assertEqual(result['decisions'][0]['action'],'retain')

    def test_invalid_timings_rejected(self):
        for bad in [0,-1,float('nan'),float('inf')]:
            p=self.probe();p['cases'][0]['timings']['wrapper_us'][0]=bad
            with self.assertRaises(ValueError):diagnose(p)

    def test_unpaired_timings_rejected(self):
        p=self.probe();p['cases'][0]['timings']['direct_us'].pop()
        with self.assertRaises(ValueError):diagnose(p)

    def test_tile_with_dispatch_noise_not_supported(self):
        p=self.probe();p['cases'][0]['tile_interventions']=[dict(status='passed',block=4096,grid=[256],
            timings=dict(parent=[100,100,100],variant=[50,100,110]))]
        result=diagnose(p)['decisions'][0]
        self.assertIsNone(result['recommended_tile'])


if __name__=='__main__':unittest.main()
