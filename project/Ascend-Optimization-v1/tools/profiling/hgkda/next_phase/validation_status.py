"""补充检查准入：新发现的语义问题不能被旧版较小测试集掩盖。"""


def supplement_status(operator, result, source_sha256):
    if result.get('source_sha256') != source_sha256:
        return 'source_mismatch'
    if result.get('status') != 'passed':
        return 'failed'
    if operator == 'negative_':
        rows = {row.get('case'): row for row in result.get('cases', [])}
        for dtype in ['float16', 'bfloat16', 'float32']:
            row = rows.get('additional-torch.' + dtype + '-signed-zero', {})
            if not row.get('passed') or 'npu_reference_signbits' not in row:
                return 'requires_target_signed_zero_check'
            if row.get('candidate_signbits') != row['npu_reference_signbits']:
                return 'failed'
    if operator == 'log10_':
        rows={row.get('case'):row for row in result.get('cases',[])}
        for dtype in ['float16','bfloat16','float32']:
            row=rows.get('additional-torch.'+dtype+'-alignment-sequence',{})
            if (not row.get('passed') or row.get('offset_sequence')!=[0,1,3,0] or row.get('elements')!=1024):
                return 'requires_same_shape_alignment_sequence'
    if operator == 'nan_to_num_':
        names={row.get('case') for row in result.get('cases',[]) if row.get('passed')}
        for dtype in ['float16','bfloat16','float32']:
            for strided in ['False','True']:
                for parameters in ['defaults','finite','nonfinite']:
                    if 'nan-parameters-torch.'+dtype+'-'+strided+'-'+parameters not in names:
                        return 'requires_nan_parameter_checks'
    if operator in {'fmod_tensor_','special_xlogy_'}:
        names={row.get('case') for row in result.get('cases',[]) if row.get('passed')}
        for dtype in ['float16','bfloat16','float32']:
            for layout in ['a-strided','b-strided','broadcast-b']:
                if 'binary-edges-torch.'+dtype+'-'+layout not in names:
                    return 'requires_binary_special_values_and_independent_layouts'
    if operator in {'hardswish','hardswish_out'}:
        names={row.get('case') for row in result.get('cases',[]) if row.get('passed')}
        for dtype in ['float16','bfloat16','float32']:
            for layout in ['flat','independent-out-strides']:
                if 'hardswish-edges-'+dtype+'-'+layout not in names:
                    return 'requires_hardswish_boundaries_range_and_out_layout'
    if operator == 'fmod_tensor_':
        names={row.get('case') for row in result.get('cases',[]) if row.get('passed')}
        for dtype in ['float16','bfloat16','float32']:
            for family in ['ordinary-boundaries','large-subnormal']:
                for layout in ['flat','independent-strides']:
                    if 'fmod-range-torch.'+dtype+'-'+family+'-'+layout not in names:
                        return 'requires_fmod_quotient_range_and_subnormal_checks'
    if operator == 'transform_bias_rescale_qkv':
        names={row.get('case') for row in result.get('cases',[]) if row.get('passed')}
        for dtype in ['float16','bfloat16','float32']:
            for layout in ['input-only','bias-only','transpose-input-bias']:
                if 'qkv-layout-torch.'+dtype+'-'+layout not in names:
                    return 'requires_independent_qkv_bias_layouts_and_output_independence'
            for pattern in ['fp16-range','nonpower7','nonpower48']:
                if 'qkv-geometry-'+dtype+'-'+pattern not in names:
                    return 'requires_qkv_head_geometry_and_pre_scale_range'
    if operator == 'cross_out':
        names={row.get('case') for row in result.get('cases',[]) if row.get('passed')}
        for dtype in ['float16','bfloat16','float32']:
            for layout in ['default-first-axis','broadcast-middle','strided-out-last','alias-out-middle','alias-out-last']:
                if 'cross-layout-torch.'+dtype+'-'+layout not in names:
                    return 'requires_cross_dimension_broadcast_and_out_checks'
    if operator == 'subtract_':
        names={row.get('case') for row in result.get('cases',[]) if row.get('passed')}
        for dtype in ['float16','bfloat16','float32']:
            for layout in ['inner-slice','transpose']:
                if 'subtract-other-layout-torch.'+dtype+'-'+layout not in names:
                    return 'requires_contiguous_destination_strided_other'
            for strided in ['False','True']:
                for alpha in ['0.0','0.5','1.0','-2.0','1.25']:
                    if 'subtract-alpha-torch.'+dtype+'-'+strided+'-'+alpha not in names:
                        return 'requires_runtime_alpha_broadcast_and_inplace_sequence'
    if operator == 'xlogy_scalar_tensor':
        names={row.get('case') for row in result.get('cases',[]) if row.get('passed')}
        for dtype in ['float16','bfloat16','float32']:
            for strided in ['False','True']:
                for scalar in ['0.0','-0.0','2.0','-3.0','inf','nan']:
                    if 'xlogy-scalars-torch.'+dtype+'-'+strided+'-'+scalar not in names:
                        return 'requires_runtime_scalar_and_special_value_sequence'
    if operator == 'hardshrink_backward':
        names={row.get('case') for row in result.get('cases',[]) if row.get('passed')}
        for dtype in ['float16','bfloat16','float32']:
            for lambd in [0.,0.5,1.25]:
                if 'hardshrink-boundary-torch.'+dtype+'-'+str(lambd) not in names:
                    return 'requires_threshold_boundary_checks'
    if operator == 'expm1_':
        rows={row.get('case'):row for row in result.get('cases',[])}
        for dtype in ['float16','bfloat16','float32']:
            row=rows.get('expm1-small-torch.'+dtype,{})
            if (not row.get('passed') or 'cpu_reference_zero_signbits' not in row or
                    not row.get('npu_contract_passed') or not row.get('cpu_small_relative_passed')):
                return 'requires_small_magnitude_relative_accuracy'
            if row.get('candidate_zero_signbits') != row['cpu_reference_zero_signbits']:
                return 'failed'
    if operator in {'round_out','special_round_out'}:
        names={row.get('case') for row in result.get('cases',[]) if row.get('passed')}
        for dtype in ['float16','bfloat16','float32']:
            for decimals in [-1,0,1,2]:
                if 'round-edges-torch.'+dtype+'-'+str(decimals) not in names:
                    return 'requires_rounding_ties_and_layout_checks'
    if operator == 'native_group_norm':
        names={row.get('case') for row in result.get('cases',[]) if row.get('passed')}
        for dtype in ['float16','bfloat16','float32']:
            for absent in ['weight','bias']:
                if 'groupnorm-optional-'+dtype+'-without-'+absent not in names:
                    return 'requires_independent_optional_affine_checks'
    if operator == 'fix':
        rows={row.get('case'):row for row in result.get('cases',[])}
        for dtype in ['float16','bfloat16','float32']:
            row=rows.get('fix-edges-torch.'+dtype,{})
            if not row.get('passed') or 'npu_reference_zero_signbits' not in row:
                return 'requires_target_truncation_edges'
            if row.get('candidate_zero_signbits') != row['npu_reference_zero_signbits']:
                return 'failed'
    if operator == 'uniform':
        names={row.get('case') for row in result.get('cases',[]) if row.get('passed')}
        required={'uniform-rng-seed_high_bits','uniform-rng-offset_high_bits','uniform-rng-state_advance_and_replay'}
        if not required.issubset(names) or not result.get('random_details'):
            return 'requires_random_state_checks'
    return 'passed'
