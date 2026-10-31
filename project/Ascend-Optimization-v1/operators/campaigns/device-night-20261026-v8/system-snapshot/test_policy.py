"""设备机制策略边界与分组调度覆盖校验，不运行NPU。"""
import unittest
from diagnose import diagnose,positive,choose_acquisition
from variants import validate,DEFAULTS


class PolicyTests(unittest.TestCase):
    def test_missing_control_retains_original(self):
        p=dict(schema='hgkda-device-interventions-v1',operator='amin',source_sha256='a'*64,
            cases=[dict(id='one',variants=[dict(name='reexpress_parent',status='failed')])])
        self.assertEqual(diagnose(p)['cases'][0]['action'],'retain')

    def test_noise_does_not_confirm_mechanism(self):
        p=dict(schema='hgkda-device-interventions-v1',operator='amin',source_sha256='a'*64,cases=[dict(id='one',spec={'shape':[64,128],'dtype':'float16'},variants=[
            dict(name='reexpress_parent',status='passed',config=DEFAULTS['amin'],timings={'variant_ms':[1,1.1,.9]}),
            dict(name='no_multibuffer',status='passed',config={**DEFAULTS['amin'],'multi':False},timings={'variant_ms':[.8,.9,1.1]})])])
        item=diagnose(p)['cases'][0]['observations'][1]
        self.assertEqual(item['status'],'not_established')

    def test_invalid_measurements_rejected(self):
        for value in [True,0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):positive([value])

    def test_grouped_tiles_cover_partial_last_group_once(self):
        for nm in [1,3,4,7,8,11]:
            for nn in [1,2,5]:
                for group in [2,4,8]:
                    coords=[]
                    for pid in range(nm*nn):
                        width=group*nn;first=(pid//width)*group;size=min(nm-first,group)
                        coords.append((first+pid%size,(pid%width)//size))
                    self.assertEqual(set(coords),{(m,n) for m in range(nm) for n in range(nn)})
                    self.assertEqual(len(coords),len(set(coords)))

    def test_invalid_config_types_rejected(self):
        for delta in [{'bm':True},{'group':-1},{'bk32':1024}]:
            with self.assertRaises(ValueError):validate('matmul_bias_activation',{**DEFAULTS['matmul_bias_activation'],**delta})

    def test_missing_evidence_does_not_trigger_simulator(self):
        item=choose_acquisition(dict(operator='amin',cases=[]),None)
        self.assertEqual(item['next_probe'],'none')

    def test_precision_effect_routes_to_ir_first(self):
        diagnosis=dict(operator='amin',cases=[dict(observations=[dict(changed={'native':{}},sample_ranges_separated=True,hypothesis='precision_expansion')])])
        item=choose_acquisition(diagnosis,{'profile_status':'completed'})
        self.assertEqual(item['next_probe'],'ir')


if __name__=='__main__':unittest.main()
