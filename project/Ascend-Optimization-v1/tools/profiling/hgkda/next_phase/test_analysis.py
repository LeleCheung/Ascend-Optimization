"""验证重叠区间、缺数据/零分母、参考身份和知识准入，避免错误性能归因。"""
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from regions import analyze,union,length,complement
from evidence import compact
from knowledge import organize,retrieve


class AnalysisTests(unittest.TestCase):
    def test_uniform_statistics_alone_do_not_qualify_generator_state(self):
        from validation_status import supplement_status
        result=dict(status='passed',source_sha256='sha',cases=[dict(case='range-and-variance',passed=True)])
        self.assertEqual(supplement_status('uniform',result,'sha'),'requires_random_state_checks')

    def test_baseline_symbol_adapter_cannot_change_computation(self):
        from baseline import verify_adapter
        original=b'def selu_(x): return x+1\n_original=selu_\ndef run(A): return _original(A)\n'
        adapted=original.replace(b'selu_',b'_fg_selu_impl')
        spec=dict(original_sha256=hashlib.sha256(original).hexdigest(),adapted_sha256=hashlib.sha256(adapted).hexdigest(),
            renamed_symbol='selu_',private_symbol='_fg_selu_impl')
        self.assertEqual(verify_adapter(original,adapted,spec)['status'],'verified_symbol_rename_only')
        changed=adapted.replace(b'x+1',b'x+2');spec['adapted_sha256']=hashlib.sha256(changed).hexdigest()
        with self.assertRaises(AssertionError):verify_adapter(original,changed,spec)

    def test_negative_requires_npu_signed_zero_checks(self):
        from validation_status import supplement_status
        result=dict(status='passed',source_sha256='sha',cases=[])
        self.assertEqual(supplement_status('negative_',result,'sha'),'requires_target_signed_zero_check')
        result['cases']=[dict(case='additional-torch.'+dtype+'-signed-zero',passed=True,
            npu_reference_signbits=[True,False],candidate_signbits=[True,False])
            for dtype in ['float16','bfloat16','float32']]
        self.assertEqual(supplement_status('negative_',result,'sha'),'passed')
        result['cases'][0]['candidate_signbits']=[False,False]
        self.assertEqual(supplement_status('negative_',result,'sha'),'failed')
        self.assertEqual(supplement_status('negative_',result,'other'),'source_mismatch')

    def test_nan_requires_layout_and_parameter_checks(self):
        from validation_status import supplement_status
        result=dict(status='passed',source_sha256='sha',cases=[])
        self.assertEqual(supplement_status('nan_to_num_',result,'sha'),'requires_nan_parameter_checks')
        result['cases']=[dict(case='nan-parameters-torch.'+dtype+'-'+strided+'-'+parameters,passed=True)
            for dtype in ['float16','bfloat16','float32'] for strided in ['False','True']
            for parameters in ['defaults','finite','nonfinite']]
        self.assertEqual(supplement_status('nan_to_num_',result,'sha'),'passed')

    def test_metric_family_does_not_treat_cube_as_ub(self):
        from evidence import metric_family
        self.assertEqual(metric_family('aic_cube_fops'),'compute')
        self.assertEqual(metric_family('aiv_vec_fp_instr_number'),'compute')
        self.assertEqual(metric_family('aiv_scalar_single_time(us)'),'control')
        self.assertEqual(metric_family('GM_to_UB_datas(KB)'),'movement')
        self.assertEqual(metric_family('aiv_read_hit_rate(%)','L2Cache.csv'),'cache')

    def test_bounded_context_keeps_ratios_and_full_evidence_identity(self):
        from context import bounded_context
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'full.json'
            metric=dict(metric='MTE2_time(us)',reference=2,candidate=4,candidate_over_reference=2,absolute_delta=2)
            value=dict(schema='test',status='completed',spec=dict(operator='x'),source=dict(path='main.py',text='x'*100000),
                comparisons=[dict(reference_kind='cann',selected_metrics=[metric],quality_flags=[dict(reason='x'*500)]*100)],
                actual_ir=dict(entries=[dict(symbol='k',artifacts=[dict(path='actual.ll',sha256='abc')])]))
            path.write_text(json.dumps(value))
            brief=bounded_context(value,path)
            self.assertLessEqual(len(json.dumps(brief,ensure_ascii=False,indent=2)),32000)
            self.assertEqual(brief['comparisons'][0]['selected_metrics'][0]['candidate_over_reference'],2)
            self.assertEqual(brief['actual_ir']['entries'][0]['artifacts'][0]['path'],'actual.ll')
            self.assertEqual(brief['full_evidence']['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())

    def test_overlap_and_capture(self):
        self.assertEqual(union([(1,5),(3,8),(8,9),(11,12)]),[[1.,9.],[11.,12.]])
        self.assertEqual(length([(1,5),(3,8)]),7)
        self.assertEqual(complement([(-1,3),(2,5),(7,15)],0,10),[[5.,7.]])

    def test_trace_wait_not_idle(self):
        listing=dict(kernels=[dict(name='k',cores=[dict(name='core0',instructions=[dict(pc='0x10',instruction='WAIT_FLAG',pipeline='VECTOR',running_time_us=3,source=dict(file='x.py',line=12))])])])
        trace=dict(traceEvents=[dict(ph='M',name='process_name',pid=1,args=dict(name='VECTOR')),dict(ph='X',pid=1,ts=0,dur=4,name='WAIT_FLAG',args=dict(pc_addr='0x10')),dict(ph='X',pid=1,ts=3,dur=3,name='ADD',args=dict(pc_addr='0x10')),dict(ph='X',pid=1,ts=8,dur=2,name='ADD',args=dict(pc_addr='0x10'))])
        d=analyze(listing,trace,core='core0',kernel='k')
        pipe=d['timeline']['pipelines'][0]
        self.assertEqual(pipe['active_union'],8)
        self.assertEqual(pipe['idle_union'],2)
        self.assertEqual(pipe['wait_union'],4)
        self.assertEqual(pipe['wait_only_union'],3)
        self.assertEqual(pipe['productive_union'],5)
        self.assertEqual(pipe['productive_union']+pipe['wait_only_union']+pipe['idle_union'],10)
        self.assertEqual(d['regions'][0]['trace_interval_union'],8)
        self.assertEqual(analyze(listing)['timeline']['status'],'unsupported')

    def test_ratio_quality_and_scope(self):
        d=compact(dict(differences=[dict(table='t',metric='mte_ratio',aggregation='mean',reference=0,candidate=.7,reference_rows=1,candidate_rows=2),dict(table='t',metric='VECTOR_time(us)',aggregation='mean',reference=2,candidate=4,reference_rows=1,candidate_rows=1)]))
        self.assertEqual(d['quality_flags'][0]['status'],'zero_reference')
        self.assertIsNone(d['quality_flags'][0]['candidate_over_reference'])
        self.assertEqual(d['selected_metrics'][1]['absolute_delta'],2)
        self.assertTrue(d['quality_flags'][0]['flags'])

    def test_no_validated_knowledge_without_bound_result(self):
        record=dict(id='r',operator_type='pointwise',action_id='tile',scope=dict(compiler='fixed'),summary='larger tile',evidence=dict(source='absent'),status='validated')
        index=organize([record,dict(record,id='old',status='draft')])
        self.assertEqual(index['active'][0]['status'],'draft')
        self.assertEqual(len(index['superseded']),1)
        self.assertEqual(retrieve(index,'pointwise',dict(compiler='fixed'))['records'],[])

    def test_reject_ambiguous_trace_binding(self):
        with self.assertRaises(ValueError):analyze(dict(kernels=[dict(name='k',cores=[])]),dict(traceEvents=[]))

    def test_auxiliary_lane_does_not_extend_kernel_window(self):
        listing=dict(kernels=[dict(name='k',cores=[dict(name='c',instructions=[dict(pc='0x10',instruction='ADD',running_time_us=2)])])])
        trace=dict(traceEvents=[dict(ph='M',name='process_name',pid=1,args=dict(name='VECTOR')),dict(ph='M',name='process_name',pid=2,args=dict(name='CACHEMISS')),
            dict(ph='X',pid=2,ts=0,dur=1,name='0x10',args=dict(pc_addr='0x10')),dict(ph='X',pid=1,ts=3,dur=2,name='ADD',args=dict(pc_addr='0x10'))])
        result=analyze(listing,trace,kernel='k',core='c')
        self.assertEqual(result['timeline']['capture_start'],3)
        self.assertEqual(result['timeline']['capture_duration'],2)
        self.assertEqual(result['regions'][0]['trace_interval_union'],2)

    def test_multikernel_metrics_add_counts_but_not_time_means(self):
        from assemble import path_metrics
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);captures=[]
            for i in [1,2]:
                path=root/str(i);path.mkdir()
                (path/'Memory.csv').write_text('block_id,sub_block_id,GM_to_UB_datas(KB),aiv_mte2_instructions,MTE2_time(us)\n0,0,'+str(i)+','+str(2*i)+','+str(10*i)+'\n')
                captures.append(dict(kernel='k'+str(i),directory=str(path)))
            result=path_metrics(captures,root/'metrics.json')
            totals={r['metric']:r for r in result['aggregate_work']}
            self.assertEqual(totals['GM_to_UB_datas(KB)']['value'],3)
            self.assertEqual(totals['aiv_mte2_instructions']['value'],6)
            self.assertNotIn('MTE2_time(us)',totals)
            (root/'2/Memory.csv').write_text('block_id,sub_block_id,aiv_mte2_instructions\n0,0,4\n')
            result=path_metrics(captures,root/'partial.json')
            total=next(r for r in result['aggregate_work'] if r['metric']=='GM_to_UB_datas(KB)')
            self.assertIsNone(total['value'])
            self.assertEqual(total['observed_sum'],1)

    def test_knowledge_requires_source_bound_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'main.py';source.write_text('def run(x): return x\n')
            formal=root/'a.result.json';formal.write_text(json.dumps(dict(status='PASSED',num_passed=2,num_workloads=2)))
            request=root/'a.request.json';request.write_text(json.dumps(dict(implementation=dict(sources=[dict(content=source.read_text())]))))
            record=dict(id='new',operator_type='pointwise',action_id='tile',scope=dict(compiler='fixed'),summary='verified implementation',status='validated',
                        evidence=dict(source='main.py',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),formal_result='a.result.json',formal_result_sha256=hashlib.sha256(formal.read_bytes()).hexdigest(),formal_request='a.request.json'))
            index=organize([record],root=root)
            self.assertEqual(len(retrieve(index,'pointwise',dict(compiler='fixed'))['records']),1)
            self.assertEqual(retrieve(index,'pointwise',dict(compiler='other'))['records'],[])
            self.assertEqual(retrieve(index,'pointwise',dict(compiler='fixed'),max_chars=1)['records'],[])
            request.write_text(json.dumps(dict(implementation=dict(sources=[dict(content='wrong code')]))))
            self.assertEqual(retrieve(index,'pointwise',dict(compiler='fixed'))['records'],[])
            self.assertEqual(organize([record],root=root)['active'][0]['status'],'draft')

    def test_knowledge_respects_new_operator_supplement_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'main.py';source.write_text('def run(x): return x\n')
            formal=root/'a.result.json';formal.write_text(json.dumps(dict(status='PASSED',num_passed=2,num_workloads=2)))
            request=root/'a.request.json';request.write_text(json.dumps(dict(implementation=dict(sources=[dict(content=source.read_text())]))))
            sha=hashlib.sha256(source.read_bytes()).hexdigest();extra=root/'extra.json'
            extra.write_text(json.dumps(dict(status='passed',operator='native_group_norm',source_sha256=sha,cases=[])))
            record=dict(id='old',operator_type='normalization',action_id='launch',scope=dict(operator='native_group_norm'),
                summary='旧的小补测集',status='validated',evidence=dict(source='main.py',source_sha256=sha,
                    formal_result='a.result.json',formal_result_sha256=hashlib.sha256(formal.read_bytes()).hexdigest(),
                    formal_request='a.request.json',supplementary=[dict(path='extra.json',sha256=hashlib.sha256(extra.read_bytes()).hexdigest())]))
            index=organize([record],root=root)
            self.assertEqual(index['active'][0]['status'],'draft')
            self.assertIn('当前算子门禁',index['active'][0]['validation_error'])

    def test_failure_constraint_needs_matching_failed_supplement(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'main.py';source.write_text('def run(x): return x\n')
            formal=root/'a.result.json';formal.write_text(json.dumps(dict(status='PASSED',num_passed=2,num_workloads=2)))
            request=root/'a.request.json';request.write_text(json.dumps(dict(implementation=dict(sources=[dict(content=source.read_text())]))))
            sha=hashlib.sha256(source.read_bytes()).hexdigest();extra=root/'extra.json'
            extra.write_text(json.dumps(dict(status='failed',source_sha256=sha)))
            record=dict(id='failure',operator_type='pointwise',action_id='signed-zero',scope=dict(compiler='fixed'),
                summary='完整合同通过但正负零补测失败；限制适用范围',status='observed_failure',
                evidence=dict(source='main.py',source_sha256=sha,formal_result='a.result.json',
                    formal_result_sha256=hashlib.sha256(formal.read_bytes()).hexdigest(),formal_request='a.request.json',
                    supplementary=[dict(path='extra.json',sha256=hashlib.sha256(extra.read_bytes()).hexdigest())]))
            index=organize([record],root=root)
            self.assertEqual(retrieve(index,'pointwise',dict(compiler='fixed'))['records'][0]['status'],'observed_failure')
            record['evidence']['supplementary']=[]
            self.assertEqual(organize([record],root=root)['active'][0]['status'],'draft')


if __name__=='__main__':unittest.main()
