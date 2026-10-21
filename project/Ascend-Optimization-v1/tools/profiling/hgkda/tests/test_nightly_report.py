"""报告不能把候选当原版，也不能把角色加载当成实际profiling闭环。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('night_report',Path(__file__).resolve().parents[1]/'report-nightly.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


class ReportTests(unittest.TestCase):
    def test_analyzer_origin_requires_verified_parent_profile(self):
        rounds=[{'round':2,'source':{'origin':'analyzer','parent_round_num':1}},
                {'round':3,'source':{'origin':'profile_next_experiment','parent_round_num':2}},
                {'round':4,'source':{'origin':'coder','parent_round_num':1}}]
        self.assertEqual(m.guided_rounds(rounds,[{'round':1}]),[rounds[0]])
        self.assertEqual(m.guided_rounds(rounds,[]),[])

    def test_partial_semantics_cannot_be_complete_by_changing_total(self):
        data={'status':'passed','operator':'value_selecting_reduction_backward',
              'source_sha256':'source','checker_sha256':'checker','physical_device':7,'logical_device':'npu:0',
              'passed':1,'total':1,'cases':[{'dtype':'fp32','case':0,'status':'passed'}]}
        self.assertFalse(m.valid_semantics(data,'value_selecting_reduction_backward','source','checker'))

    def profile_fixture(self,root,failed_case='case'):
        entries=[]
        files={
            '.kernelgen/evals/round-0001/identity.json':{'solution_sha256':'solution'},
            '.kernelgen/profile-analysis/round-0001.json':{
                'status':'completed','findings':[{'evidence':'metrics'}],'next_experiment':{'change':'tile'},
                'round_num':1,'solution_sha256':'solution',
                'profiled_workloads':[{'status':'completed','uuid':'case','profile_ids':['metrics','instruction']}]},
        }
        for key,status,case in [('metrics','completed','case'),('instruction','failed',failed_case)]:
            files['.kernelgen/profiles/'+key+'/manifest.json']={
                'request':{'identity':{'solution_sha256':'solution'},'case_id':case},
                'result':{'profile_id':key,'status':status,'artifacts':[{'kind':'profile_log'}],
                          'error':'deadline expired' if status=='failed' else None}}
        for index,(name,value) in enumerate(files.items()):
            path=root/'native-evidence/files'/('%d.json'%index);save(path,value)
            entries.append({'source_path':name,'archived_path':'files/'+path.name,
                            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        save(root/'native-evidence/archive-manifest.json',{'files':entries})

    def test_failed_instruction_receipt_is_preserved_without_counting_as_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.profile_fixture(root)
            data=m.check_profile_evidence(root)
            self.assertEqual(data['successful_profiles'],1)
            self.assertEqual(data['verified_completed_analyses'][0]['profile_ids'],['metrics'])
            self.assertEqual(data['failed_profiles'][0]['error'],'deadline expired')

    def test_failed_profile_still_requires_matching_case_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.profile_fixture(root,failed_case='wrong-case')
            with self.assertRaisesRegex(ValueError,'case不一致'):m.check_profile_evidence(root)

    def fixture(self,batch):
        records=[]
        for op in ['value_selecting_reduction_backward','moe_sum_reduce']:
            records.append({'operator':op,'source':'flaggems_sglang','coverage':'test',
                            'source_sha256':'base','catalog_leaf':'catalog/'+op})
            root=batch/'results'/op
            for arm in ['kg-no-profile','kg-profile','HG-KDA']:
                for i in [1,2]:save(root/arm/'independent'/(arm+'-%d.result.json'%i),{'status':'PASSED'})
                if arm=='HG-KDA':
                    role=b'# HG-KDA '+ '角色扩展'.encode()
                    (root/arm/'role.md').write_bytes(role)
                    save(root/arm/'role-load-evidence.json',{'hgkda_roles_found':[{'archived':'role.md','sha256':hashlib.sha256(role).hexdigest()}]})
        save(batch/'plan.json',{'records':records})
        save(batch/'queue-state.json',{'status':'completed'})

    def loaded(self,path,*unused):
        return {'path':str(path),'source_sha256':'candidate','binding':{},'fingerprint':'same',
                'device':'npu:0','timing_scope':'walltime','correctness_count':1,
                'result_sha256':'result','rows':{'a':{'phase':'correctness'},'b':{'phase':'timing'}},
                'timing':{'b':{'reference_latency_ms':2.,'latency_ms':1.}}}

    def test_missing_baseline_does_not_become_candidate_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            batch=Path(temp);self.fixture(batch)
            with patch.object(m,'verify',side_effect=self.loaded):m.build(batch)
            result=m.read(batch/'统计核验.json')
            self.assertFalse(result['all_priority_repeats_and_integration_verified'])
            for value in result['priority_comparisons'].values():
                for row in value['versions']:
                    self.assertIsNone(row['relative_master'])
                    self.assertEqual(row['relative_pytorch'],2.)
                self.assertFalse(value['groups']['HG-KDA']['integration_verified'])

    def test_host_loop_reference_is_not_counted_as_native_kernel(self):
        with tempfile.TemporaryDirectory() as temp:
            batch=Path(temp);self.fixture(batch)
            plan=m.read(batch/'plan.json')
            plan['records'].append({'operator':'silu_and_mul_masked','source':'flaggems_sglang',
                'coverage':'test','source_sha256':'candidate','catalog_leaf':'catalog/silu'})
            save(batch/'plan.json',plan)
            save(batch/'results/silu_and_mul_masked/FlagGems/FlagGems-1.result.json',{'status':'PASSED'})
            save(batch/'reference-execution-audit.json',{'silu_and_mul_masked':{
                'cpu_fallback':False,'reference_scope':'compound_pytorch_with_host_loop'}})
            with patch.object(m,'verify',side_effect=self.loaded):summary=m.build(batch)
            self.assertEqual(summary['native_reference_scores'],0)
            self.assertEqual(summary['compound_reference_scores'],1)
            row=m.read(batch/'统计核验.json')['baseline_table'][-1]
            self.assertEqual(row['score'],2.)
            self.assertEqual(row['reference_scope'],'compound_pytorch_with_host_loop')
            self.assertIn('非新增优化收益',(batch/'结果汇总.md').read_text(encoding='utf-8'))

    def test_require_complete_rejects_loaded_role_without_actual_analysis(self):
        with tempfile.TemporaryDirectory() as temp:
            batch=Path(temp);self.fixture(batch)
            with patch.object(m,'verify',side_effect=self.loaded):
                with self.assertRaisesRegex(ValueError,'实际集成证据'):m.build(batch,True)
            self.assertFalse(m.read(batch/'统计核验.json')['all_priority_repeats_and_integration_verified'])
            self.assertTrue((batch/'结果汇总.md').is_file())

    def test_contract_byte_drift_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            leaf=Path(temp);(leaf/'oracle.py').write_bytes(b'changed\n')
            record={'contract_sha256':{'oracle.py':hashlib.sha256(b'original\n').hexdigest()}}
            with self.assertRaisesRegex(ValueError,'冻结合同'):m.verify(leaf/'unused',record,leaf)

    def test_adapter_retest_preserves_upstream_and_workload_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            parent=Path(temp);batch=parent/'original';retry=parent/'retry'
            record={'operator':'silu','catalog_leaf':'catalog/ops/silu','upstream_commit':'fixed',
                    'upstream_source_sha256':'original-code'}
            for root in [batch,retry]:
                leaf=root/record['catalog_leaf'];leaf.mkdir(parents=True)
                for name in ['definition.json','correctness.jsonl','timing.jsonl']:(leaf/name).write_bytes(b'fixed\n')
            plan=retry/'plan.json';save(plan,{'records':[record]})
            result=retry/'results/silu/FlagGems/FlagGems-1.result.json';save(result,{'status':'PASSED'})
            issue={'retest_batch':'retry','retest_plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest()}
            self.assertTrue(m.baseline_location(batch,record,issue)[3])
            (retry/record['catalog_leaf']/'timing.jsonl').write_bytes(b'reduced case scope\n')
            with self.assertRaisesRegex(ValueError,'ABI或用例'):m.baseline_location(batch,record,issue)


if __name__=='__main__':unittest.main()
