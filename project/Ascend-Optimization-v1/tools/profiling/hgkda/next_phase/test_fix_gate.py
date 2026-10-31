"""截断修复必须有目标 NPU 的边界与正负零证据。"""
import unittest
from pathlib import Path

from validation_status import supplement_status


class FixGateTests(unittest.TestCase):
    def test_hardswish_precision_changes_need_boundaries_and_range(self):
        result=dict(status='passed',source_sha256='source',cases=[])
        for op in ['hardswish','hardswish_out']:
            self.assertEqual(supplement_status(op,result,'source'),'requires_hardswish_boundaries_range_and_out_layout')
        result['cases']=[dict(case='hardswish-edges-'+dtype+'-'+layout,passed=True)
                         for dtype in ['float16','bfloat16','float32'] for layout in ['flat','independent-out-strides']]
        for op in ['hardswish','hardswish_out']:
            self.assertEqual(supplement_status(op,result,'source'),'passed')

    def test_log10_cached_launch_needs_same_shape_alignment_changes(self):
        result=dict(status='passed',source_sha256='source',cases=[])
        self.assertEqual(supplement_status('log10_',result,'source'),'requires_same_shape_alignment_sequence')
        result['cases']=[dict(case='additional-torch.'+dtype+'-alignment-sequence',passed=True,
                             offset_sequence=[0,1,3,0],elements=1024)
                         for dtype in ['float16','bfloat16','float32']]
        self.assertEqual(supplement_status('log10_',result,'source'),'passed')

    def test_binary_inplace_operators_need_both_operand_layouts_and_edges(self):
        result=dict(status='passed',source_sha256='source',cases=[])
        for operator in ['fmod_tensor_','special_xlogy_']:
            self.assertEqual(supplement_status(operator,result,'source'),
                             'requires_binary_special_values_and_independent_layouts')
        result['cases']=[dict(case='binary-edges-torch.'+dtype+'-'+layout,passed=True)
                         for dtype in ['float16','bfloat16','float32']
                         for layout in ['a-strided','b-strided','broadcast-b']]
        self.assertEqual(supplement_status('special_xlogy_',result,'source'),'passed')
        self.assertEqual(supplement_status('fmod_tensor_',result,'source'),
                         'requires_fmod_quotient_range_and_subnormal_checks')

    def test_fmod_needs_broad_quotient_range_and_subnormal_inputs(self):
        rows=[dict(case='binary-edges-torch.'+dtype+'-'+layout,passed=True)
              for dtype in ['float16','bfloat16','float32']
              for layout in ['a-strided','b-strided','broadcast-b']]
        result=dict(status='passed',source_sha256='source',cases=rows)
        self.assertEqual(supplement_status('fmod_tensor_',result,'source'),
                         'requires_fmod_quotient_range_and_subnormal_checks')
        rows.extend(dict(case='fmod-range-torch.'+dtype+'-'+family+'-'+layout,passed=True)
                    for dtype in ['float16','bfloat16','float32']
                    for family in ['ordinary-boundaries','large-subnormal']
                    for layout in ['flat','independent-strides'])
        self.assertEqual(supplement_status('fmod_tensor_',result,'source'),'passed')
        rows[-1]['passed']=False
        self.assertEqual(supplement_status('fmod_tensor_',result,'source'),
                         'requires_fmod_quotient_range_and_subnormal_checks')

    def test_qkv_needs_independent_operand_layouts_and_output_independence(self):
        result=dict(status='passed',source_sha256='source',cases=[])
        self.assertEqual(supplement_status('transform_bias_rescale_qkv',result,'source'),
                         'requires_independent_qkv_bias_layouts_and_output_independence')
        result['cases']=[dict(case='qkv-layout-torch.'+dtype+'-'+layout,passed=True)
                         for dtype in ['float16','bfloat16','float32']
                         for layout in ['input-only','bias-only','transpose-input-bias']]
        self.assertEqual(supplement_status('transform_bias_rescale_qkv',result,'source'),
                         'requires_qkv_head_geometry_and_pre_scale_range')
        result['cases'].extend(dict(case='qkv-geometry-'+dtype+'-'+pattern,passed=True)
                               for dtype in ['float16','bfloat16','float32']
                               for pattern in ['fp16-range','nonpower7','nonpower48'])
        self.assertEqual(supplement_status('transform_bias_rescale_qkv',result,'source'),'passed')

    def test_subtract_needs_runtime_alpha_and_independent_operand_layouts(self):
        rows=[dict(case='subtract-alpha-torch.'+dtype+'-'+strided+'-'+alpha,passed=True)
              for dtype in ['float16','bfloat16','float32'] for strided in ['False','True']
              for alpha in ['0.0','0.5','1.0','-2.0','1.25']]
        result=dict(status='passed',source_sha256='source',cases=rows)
        self.assertEqual(supplement_status('subtract_',result,'source'),
                         'requires_contiguous_destination_strided_other')
        rows.extend(dict(case='subtract-other-layout-torch.'+dtype+'-'+layout,passed=True)
                    for dtype in ['float16','bfloat16','float32'] for layout in ['inner-slice','transpose'])
        self.assertEqual(supplement_status('subtract_',result,'source'),'passed')
        rows.pop(0)
        self.assertEqual(supplement_status('subtract_',result,'source'),
                         'requires_runtime_alpha_broadcast_and_inplace_sequence')

    def test_xlogy_requires_reused_storage_scalar_sequence(self):
        result=dict(status='passed',source_sha256='source',cases=[])
        self.assertEqual(supplement_status('xlogy_scalar_tensor',result,'source'),
                         'requires_runtime_scalar_and_special_value_sequence')
        result['cases']=[dict(case='xlogy-scalars-torch.'+dtype+'-'+strided+'-'+scalar,passed=True)
                         for dtype in ['float16','bfloat16','float32'] for strided in ['False','True']
                         for scalar in ['0.0','-0.0','2.0','-3.0','inf','nan']]
        self.assertEqual(supplement_status('xlogy_scalar_tensor',result,'source'),'passed')

    def test_supplement_revisions_keep_old_results_and_reject_unsafe_paths(self):
        from proposals import extra_gate_path
        root=Path('assessment')
        self.assertEqual(extra_gate_path(root,{}),(root/'extra-checks.json',''))
        path,suffix=extra_gate_path(root,dict(extra_check_revision='reference-policy-v2'))
        self.assertEqual(path,root/'extra-checks-reference-policy-v2.json')
        self.assertEqual(suffix,':reference-policy-v2')
        for revision in ['../old','/tmp/old','bad/name','',3]:
            with self.assertRaises(ValueError):extra_gate_path(root,dict(extra_check_revision=revision))

    def test_hardshrink_requires_threshold_neighbors_for_each_dtype(self):
        result=dict(status='passed',source_sha256='source',cases=[])
        self.assertEqual(supplement_status('hardshrink_backward',result,'source'),'requires_threshold_boundary_checks')
        result['cases']=[dict(case='hardshrink-boundary-torch.'+dtype+'-'+str(lambd),passed=True)
                         for dtype in ['float16','bfloat16','float32'] for lambd in [0.,0.5,1.25]]
        self.assertEqual(supplement_status('hardshrink_backward',result,'source'),'passed')

    def test_expm1_requires_relative_small_value_and_zero_sign_evidence(self):
        result=dict(status='passed',source_sha256='source',cases=[])
        self.assertEqual(supplement_status('expm1_',result,'source'),'requires_small_magnitude_relative_accuracy')
        result['cases']=[dict(case='expm1-small-torch.'+dtype,passed=True,
                             npu_reference_zero_signbits=[False,False],cpu_reference_zero_signbits=[False,True],
                             candidate_zero_signbits=[False,True],npu_contract_passed=True,cpu_small_relative_passed=True)
                         for dtype in ['float16','bfloat16','float32']]
        self.assertEqual(supplement_status('expm1_',result,'source'),'passed')

    def test_round_interfaces_each_require_tie_and_layout_evidence(self):
        result=dict(status='passed',source_sha256='source',cases=[])
        for op in ['round_out','special_round_out']:
            self.assertEqual(supplement_status(op,result,'source'),'requires_rounding_ties_and_layout_checks')
        result['cases']=[dict(case='round-edges-torch.'+dtype+'-'+str(decimals),passed=True)
                         for dtype in ['float16','bfloat16','float32'] for decimals in [-1,0,1,2]]
        for op in ['round_out','special_round_out']:
            self.assertEqual(supplement_status(op,result,'source'),'passed')

    def test_groupnorm_both_or_neither_is_insufficient(self):
        result=dict(status='passed',source_sha256='source',cases=[])
        self.assertEqual(supplement_status('native_group_norm',result,'source'),
                         'requires_independent_optional_affine_checks')
        result['cases']=[dict(case='groupnorm-optional-'+dtype+'-without-'+absent,passed=True)
                         for dtype in ['float16','bfloat16','float32'] for absent in ['weight','bias']]
        self.assertEqual(supplement_status('native_group_norm',result,'source'),'passed')

    def test_old_generic_suite_cannot_qualify_fix(self):
        result=dict(status='passed',source_sha256='source',cases=[])
        self.assertEqual(supplement_status('fix',result,'source'),'requires_target_truncation_edges')

    def test_target_zero_sign_must_match(self):
        rows=[dict(case='fix-edges-torch.'+dtype,passed=True,
                   npu_reference_zero_signbits=[False,True],candidate_zero_signbits=[False,True])
              for dtype in ['float16','bfloat16','float32']]
        result=dict(status='passed',source_sha256='source',cases=rows)
        self.assertEqual(supplement_status('fix',result,'source'),'passed')
        rows[1]['candidate_zero_signbits']=[False,False]
        self.assertEqual(supplement_status('fix',result,'source'),'failed')


if __name__=='__main__':unittest.main()
