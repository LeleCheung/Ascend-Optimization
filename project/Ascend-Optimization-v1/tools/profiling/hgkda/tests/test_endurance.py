"""控制器结果选择遵循实际耗时，且准备器保持原合同字节。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import hashlib
import io
import tarfile
from types import SimpleNamespace

TOOLS=Path(__file__).resolve().parents[1]


def load(name,filename):
    s=importlib.util.spec_from_file_location(name,TOOLS/filename)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


class EnduranceTests(unittest.TestCase):
    def test_selection_audit_rejects_faster_reference_score_and_changed_cases(self):
        audit=load('endurance_selection_audit','audit-endurance.py')
        def measured(digest,score,latencies):
            return dict(source_sha256=digest,result=dict(status='PASSED',geo_mean=score,
                per_workload=[dict(uuid=case,phase='timing',latency_ms=value) for case,value in latencies.items()]))
        fast=measured('fast',.8,{'small':1.,'large':2.})
        slow=measured('slow',1.5,{'small':2.,'large':3.})
        result=audit.verify_fastest('fast',[fast,slow,fast],['small','large'])
        self.assertEqual(result['compared_sources'],2)
        with self.assertRaisesRegex(ValueError,'实际耗时最低'):
            audit.verify_fastest('slow',[fast,slow],['small','large'])
        for rows in [{'small':1.,'different':2.},{'small':float('nan'),'large':2.}]:
            with self.assertRaises(ValueError):
                audit.verify_fastest('bad',[measured('bad',2.,rows)],['small','large'])
        with self.assertRaisesRegex(ValueError,'重复测量挑分'):
            audit.verify_fastest('fast',[fast,measured('fast',3.,{'small':.5,'large':1.})],['small','large'])

    def test_debug_gate_bundle_keeps_actual_contract_and_detects_tampering(self):
        prepare=load('prepare_debug_bundle','prepare-endurance.py')
        staging=load('stage_debug_bundle','stage-endurance-check.py')
        checker=load('check_debug_bundle','check-endurance.py')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);batch=root/'batch';prepare.prepare(batch)
            # 实际批次的plan曾在Windows生成；元数据统一换行必须显式登记两个SHA。
            plan=batch/'plan.json';plan.write_bytes(plan.read_bytes().replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'))
            workspace=root/'workspace';source=workspace/'tmp/main.py';source.parent.mkdir(parents=True)
            source.write_bytes(b'def run(y, dx=1, dim=-1): return y\n')
            out=workspace/'tmp/independent-check'
            result=staging.stage(batch,workspace,source,out,'cumulative_trapezoid')
            self.assertEqual((out/'candidate/main.py').read_bytes(),source.read_bytes())
            self.assertEqual(json.loads((out/'plan.json').read_bytes()),json.loads(plan.read_bytes()))
            manifest=json.loads((out/'bundle-manifest.json').read_bytes())
            self.assertTrue(manifest['plan_newline_normalized'])
            self.assertEqual(manifest['plan_sha256'],staging.sha(plan.read_bytes()))
            self.assertEqual(manifest['bundled_plan_sha256'],staging.sha((out/'plan.json').read_bytes()))
            leaf,rows=checker.native_timing_contract(out/'candidate/main.py','cumulative_trapezoid')
            self.assertEqual(len(rows),9)
            self.assertEqual((leaf/'oracle.py').read_bytes(),(batch/'catalog/ops/cumulative_trapezoid/oracle.py').read_bytes())
            for entry in result['debug_request']['files']:
                self.assertTrue(entry['path'].startswith('tmp/'))
                self.assertTrue((workspace/entry['path']).is_file())
            (leaf/'oracle.py').write_bytes(b'changed\n')
            with self.assertRaises(ValueError):checker.native_timing_contract(out/'candidate/main.py','cumulative_trapezoid')
            with self.assertRaises(ValueError):staging.stage(batch,workspace,source,root/'outside','cumulative_trapezoid')
            source.write_bytes(b'def run(y):\r\n    return y\r\n')
            rejected=workspace/'tmp/crlf-check'
            with self.assertRaisesRegex(ValueError,'换行字节'):
                staging.stage(batch,workspace,source,rejected,'cumulative_trapezoid')
            self.assertFalse(rejected.exists())

    def test_ir_capture_reaches_wrapped_jit_without_recompiling(self):
        m=load('endurance_ir','observe-endurance.py')
        compiled=SimpleNamespace(asm={'ttir':'actual compiled IR'})
        jit=SimpleNamespace(device_caches={0:({'first':compiled},)})
        wrapper=SimpleNamespace(fn=SimpleNamespace(fn=jit))
        module=SimpleNamespace(kernel=wrapper,alias=wrapper)
        rows=list(m.compiled_kernels(module))
        self.assertEqual(len(rows),1)
        self.assertIs(rows[0][2],compiled)
        self.assertEqual(rows[0][3],2)
        wrapper.fn=wrapper
        self.assertEqual(list(m.compiled_kernels(module)),[])

    def test_rng_gate_rejects_truncated_keys_and_restores_state(self):
        m=load('endurance_rng_gate','check-endurance.py')
        class Generator:
            seed=77
            offset=128
            def get_state(self):return self.seed,self.offset
            def set_state(self,state):self.seed,self.offset=state
            def manual_seed(self,seed):self.seed,self.offset=seed,0
            def set_offset(self,offset):self.offset=offset
            def get_offset(self):return self.offset
        for mode in ['honest','truncated','detached']:
            gen=Generator();saved=gen.get_state()
            detached=[0]
            def run(*args):
                if mode=='detached':
                    result=(gen.seed,detached[0]);detached[0]+=4096;return result
                seed,offset=gen.get_state();gen.offset+=4096
                return (seed&0x7fffffff,offset&0x7fffffff) if mode=='truncated' else (seed,offset)
            torch=SimpleNamespace(float32=object(),empty=lambda *a,**k:None,equal=lambda a,b:a==b,
                npu=SimpleNamespace(default_generators=[gen],current_device=lambda:0,synchronize=lambda:None))
            result=m.uniform_generator_checks(SimpleNamespace(run=run),torch)
            self.assertEqual(result['passed'],3 if mode=='honest' else 1)
            self.assertEqual(gen.get_state(),saved)

    def test_scan_timing_gate_uses_frozen_large_inputs(self):
        m=load('endurance_semantics','check-endurance.py')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);leaf=root/'catalog/ops/cumulative_trapezoid';leaf.mkdir(parents=True)
            files={'oracle.py':b'def run(y): return y\n',
                   'timing.jsonl':b'{"name":"large-dot","inputs":{"spec":{"shape":[1024,4096]}}}\n'}
            for name,data in files.items():(leaf/name).write_bytes(data)
            record=dict(operator='cumulative_trapezoid',catalog_leaf='catalog/ops/cumulative_trapezoid',timing_cases=1,
                contract_sha256={n:hashlib.sha256(d).hexdigest() for n,d in files.items()})
            (root/'plan.json').write_text(json.dumps(dict(records=[record])))
            source=root/'results/candidate/main.py';source.parent.mkdir(parents=True);source.write_text('')
            observed,rows=m.native_timing_contract(source,'cumulative_trapezoid')
            self.assertEqual(observed,leaf);self.assertEqual(rows[0]['inputs']['spec']['shape'],[1024,4096])
            (leaf/'timing.jsonl').write_text('')
            with self.assertRaises(ValueError):m.native_timing_contract(source,'cumulative_trapezoid')

    def test_uploaded_contract_and_individual_cases_are_verified(self):
        m=load('endurance_audit','audit-endurance.py')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            contents={'definition.json':b'{}','oracle.py':b'def run(A): return A\n',
                'correctness.jsonl':b'{"name":"check-1"}\n','timing.jsonl':b'{"name":"time-1"}\n'}
            with tarfile.open(root/'operator-bundle.tar','w') as archive:
                for name,data in contents.items():
                    item=tarfile.TarInfo(name);item.size=len(data);archive.addfile(item,io.BytesIO(data))
            record=dict(contract_sha256={k:hashlib.sha256(v).hexdigest() for k,v in contents.items()})
            (root/'inspect.json').write_text(json.dumps(dict(benchmark_fingerprint='fixed-contract')))
            (root/'once.request.json').write_text(json.dumps(dict(binding=dict(bundle_id='sha256:'+m.sha(root/'operator-bundle.tar')))))
            result=dict(benchmark_fingerprint='fixed-contract',per_workload=[dict(uuid='check-1',phase='correctness',status='PASSED'),
                dict(uuid='time-1',phase='timing',status='PASSED')])
            m.verify_formal_contract(record,root,result)
            result['per_workload'][1]['uuid']='different-case'
            with self.assertRaises(ValueError):m.verify_formal_contract(record,root,result)
            result['per_workload'][1]['uuid']='time-1'
            record['contract_sha256']['oracle.py']='0'*64
            with self.assertRaises(ValueError):m.verify_formal_contract(record,root,result)

    def test_actual_latency_ranking_ignores_reference_drift(self):
        m=load('endurance_logic','run-endurance.py')
        parent=dict(geo_mean=.75,per_workload=[dict(uuid='a',phase='timing',latency_ms=.020),dict(uuid='b',phase='timing',latency_ms=.080)])
        faster=dict(geo_mean=.70,per_workload=[dict(uuid='b',phase='timing',latency_ms=.040),dict(uuid='a',phase='timing',latency_ms=.010)])
        self.assertAlmostEqual(m.improved(parent,faster),2.)
        self.assertLess(m.latency(faster),m.latency(parent))
        faster['per_workload'][0]['uuid']='foreign'
        with self.assertRaises(ValueError):m.improved(parent,faster)

    def test_prepare_keeps_contract_and_latest_selu(self):
        m=load('endurance_prepare','prepare-endurance.py')
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'new-batch';plan=m.prepare(out)
            self.assertEqual([r['operator'] for r in plan['records']],m.ORDER)
            self.assertEqual(plan['final_repeats'],1)
            previous=m.PROJECT/'operators/campaigns/night-20261021-v3'
            for row in plan['records']:
                leaf=Path(row['catalog_leaf'])
                for filename in row['contract_sha256']:
                    self.assertEqual((out/leaf/filename).read_bytes(),(previous/leaf/filename).read_bytes())
            self.assertEqual(plan['records'][0]['source_sha256'],'49bdbb789d497b6423f893f3433167f3fd4bc61197676236bd62a9ec79c7dd91')
            with self.assertRaises(FileExistsError):m.prepare(out)


if __name__=='__main__':unittest.main()
