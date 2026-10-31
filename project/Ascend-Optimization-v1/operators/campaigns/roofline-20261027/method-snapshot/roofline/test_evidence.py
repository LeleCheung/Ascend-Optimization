"""以真机归档和破坏性反例核验数值口径，离线运行不占NPU。"""
import copy,hashlib,importlib.util,json,math,os,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3]
sys.path.insert(0,str(HERE))
from profile_data import parse,visual_sections
from analyze import derive,load_reference
from summarize import compact
BATCH=PROJECT/'operators/campaigns/roofline-20261027'

class EvidenceTests(unittest.TestCase):
    def test_macro_counter_is_not_fixed128_correction(self):
        for name,gap in [('gemm-256-t64-k32',32),('gemm-512-t128-k32',128),('gemm-512-t128-k64',256)]:
            directory=BATCH/'calibration'/name;facts=parse(directory/'profile');spec=json.loads((directory/'spec.json').read_bytes())
            expected=2*math.prod(spec['shape'])
            self.assertEqual(expected/facts['csv_cube_fops'],gap)
            self.assertEqual(facts['csv_cube_fops'],8192*facts['csv_cube_instructions'])
            self.assertEqual(2*math.prod(spec['tile'])*facts['csv_cube_instructions'],expected)
            cube=next(c for c in facts['curves'] if c['component']=='Memory Unit(Cube)')
            self.assertAlmostEqual(cube['official_flops']/expected,1,places=5)

    def test_precision_and_component_work_are_separate(self):
        empirical=load_reference(BATCH)
        for dt,factor in [('bfloat16',1),('float32',4)]:
            facts=parse(BATCH/'calibration-v2'/('gemm-bf16' if dt=='bfloat16' else 'gemm-fp32')/'profile')
            cube=next(c for c in facts['curves'] if c['component']=='Memory Unit(Cube)')
            self.assertAlmostEqual(cube['official_flops']/(2*512**3),factor,places=4)
        facts=parse(BATCH/'matrix-profile/parent-float16-1024/profile')
        result=derive(facts,[1024]*3,'float16',empirical)
        vectors=[c for c in result['curves'] if 'Vector' in c['component']]
        self.assertTrue(vectors)
        self.assertTrue(all('algorithmic_flops' not in c for c in vectors))

    def test_gm_does_not_equal_memory_csv_interface(self):
        facts=parse(BATCH/'matrix-profile/parent-float16-1024/profile')
        gm=next(c for c in facts['curves'] if c['path']=='GM Read + Write')
        interface=sum(v for k,v in facts['memory_csv_bytes'].items() if 'main_memory' in k and v is not None)
        self.assertGreater(interface,gm['official_bytes']*2)

    def test_empty_copy_is_not_zero_roofline(self):
        facts=parse(BATCH/'calibration/copy-large/profile')
        self.assertEqual(facts['curves'],[])
        empirical=load_reference(BATCH)
        for shape,dt in [([0,512,512],'float16'),([512]*3,'copy')]:
            with self.assertRaises(ValueError):derive(facts,shape,dt,empirical)

    def test_corrupt_visual_sections_rejected(self):
        original=next((BATCH/'matrix-profile/parent-float16-1024/profile').rglob('visualize_data.bin')).read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)/'visual.bin'
            for raw in [original[:-1],b'bad',b'\xff'*12]:
                p.write_bytes(raw)
                with self.assertRaises((ValueError,UnicodeDecodeError)):visual_sections(p)

    def test_identity_mismatch_rejected(self):
        import shutil
        with tempfile.TemporaryDirectory() as temporary:
            target=Path(temporary)
            for name in ['OpBasicInfo.csv','visualize_data.bin']:
                source=next((BATCH/'matrix-profile/parent-float16-1024/profile').rglob(name));shutil.copyfile(source,target/name)
            p=target/'OpBasicInfo.csv';p.write_text(p.read_text(encoding='utf8').replace('36.280727','99.000000'),encoding='utf8')
            with self.assertRaisesRegex(ValueError,'身份'):parse(target)

    def test_nonfinite_reference_rejected(self):
        facts=parse(BATCH/'matrix-profile/parent-float16-1024/profile')
        empirical=load_reference(BATCH)
        for value in [0,float('nan'),float('inf')]:
            broken=copy.deepcopy(empirical)
            next(r for r in broken['records'] if r['kind']=='gemm' and r['dtype']=='float16')['tflops']=value
            with self.assertRaises(ValueError):derive(facts,[1024]*3,'float16',broken)

    def test_summary_binds_raw_analysis(self):
        path=BATCH/'analysis-parent-v3/diagnosis.json';summary=compact(path)
        self.assertEqual(len(summary['cases']),8)
        self.assertEqual(summary['source']['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertTrue(all(len(c['paths'])==2 for c in summary['cases']))

    def test_gm_reference_uses_same_counter_model(self):
        empirical=load_reference(BATCH);ref=empirical['gm_counter_reference']
        self.assertNotAlmostEqual(ref['gbps'],ref['logical_gbps'],places=1)
        self.assertAlmostEqual(ref['gbps'],ref['bytes']/ref['duration_us']/1000)
        broken=copy.deepcopy(empirical);broken.pop('gm_counter_reference')
        facts=parse(BATCH/'matrix-profile/parent-float16-1024/profile')
        with self.assertRaises(ValueError):derive(facts,[1024]*3,'float16',broken)

    def test_final_summary_and_pipeline_bind_real_data(self):
        path=BATCH/'analysis-final/diagnosis.json';summary=compact(path)
        self.assertEqual(len(summary['cases']),12)
        self.assertEqual(summary['source']['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
        for item in summary['cases']:
            self.assertIn('aic_cube_ratio',item['pipeline'])
        data=json.loads(path.read_bytes())
        pair={r['spec']['side']:r for r in data['cases'] if r['spec']['shape']==[1024]*3 and r['spec']['dtype']=='float16'}
        def traffic(side,path):return next(c['official_bytes'] for c in pair[side]['facts']['curves'] if c['path']==path)
        self.assertAlmostEqual(traffic('candidate','GM Read + Write')/traffic('parent','GM Read + Write'),1,places=5)
        self.assertGreater(traffic('candidate','L2 Read + Write')/traffic('parent','L2 Read + Write'),1.2)

if __name__=='__main__':unittest.main()
