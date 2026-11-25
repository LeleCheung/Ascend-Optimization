"""Plan staged diagnostics from operator types and hardware-pipeline evidence; formal evaluation remains separate."""


from __future__ import annotations


import argparse


import json


import math


from pathlib import Path


TYPES = {
    'selu_': 'pointwise', 'rad2deg_': 'pointwise', 'log10_': 'pointwise',
    '_conj_physical': 'pointwise', 'logaddexp2': 'pointwise',
    'uniform': 'random', 'cumulative_trapezoid': 'scan',
    'amin': 'reduction', 'moe_sum_reduce': 'reduction',
    'per_token_quant_int8': 'reduction',
    'value_selecting_reduction_backward': 'scatter', 'scatter_add': 'scatter',
    'upsample_bilinear2d': 'sampling', 'fractional_max_pool3d': 'sampling',
    'rotary_embedding': 'indexed_pointwise',
    'gelu_and_mul': 'fused_pointwise', 'silu_and_mul_masked': 'fused_pointwise',
    'matmul_bias_activation': 'gemm', 'narrow_copy': 'copy',
    'negative_': 'pointwise', 'xlogy_scalar_tensor': 'pointwise',
    'nan_to_num_': 'pointwise', 'cross_out': 'indexed_pointwise',
    'expm1_': 'pointwise', 'hardshrink_backward': 'pointwise',
    'subtract_': 'pointwise', 'native_group_norm': 'reduction',
    'log2_': 'pointwise', 'transform_bias_rescale_qkv': 'fused_pointwise',
    'fix': 'pointwise', 'round_out': 'pointwise', 'hardswish_out': 'pointwise',
    'fake_quantize_per_tensor_affine_cachemask_out': 'fused_pointwise',
    'special_round_out': 'pointwise', 'special_xlogy_': 'pointwise',
    'hardswish': 'pointwise', 'fmod_tensor_': 'pointwise',
    'linalg_inv_ex': 'gemm', 'batch_norm_with_update_functional': 'reduction',
}


KNOWLEDGE = {kind: 'operators.md' for kind in set(TYPES.values())}


METRICS = {
    'aiv_mte2_ratio': ('data_movement', 'Inspect contiguity, block size, and repeated input loads'),
    'aiv_mte3_ratio': ('data_movement', 'Inspect output movement, write granularity, and temporary results'),
    'aic_mte2_ratio': ('data_movement', 'Inspect repeated matrix-panel loads, K tiling, and compute overlap'),
    'aic_mte1_ratio': ('data_movement', 'Inspect on-chip matrix-panel movement and reuse'),
    'aiv_vec_ratio': ('vector_compute', 'Inspect transcendental functions, type conversions, and reduction instructions'),
    'aic_mac_ratio': ('cube_compute', 'Inspect matrix-multiply efficiency, tiles, and accumulation paths'),
    'aiv_scalar_ratio': ('control', 'Inspect address computation, loops, and program coverage'),
    'aic_scalar_ratio': ('control', 'Inspect matrix-multiply loops, indexing, and core coverage'),
}


def diagnose_case(case: dict, *, levels=('metrics',), capabilities=()) -> dict:
    operator = case['operator']
    kind = TYPES.get(operator, 'unknown')
    metrics = case.get('metrics') or {}
    hypotheses, invalid, valid_count = [], [], 0
    for name, (category, action) in METRICS.items():
        value = metrics.get(name)
        if value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            invalid.append(name)
            continue
        valid_count += 1
        threshold = 0.5 if category == 'control' else 0.7
        if value >= threshold:
            hypotheses.append({'category': category, 'metric': name, 'value': value,
                               'threshold': threshold, 'confidence': 'medium',
                               'action': action,
                               'interpretation': 'Pipeline activity prioritizes this investigation but does not establish a total-latency bottleneck'})
    profiles = case.get('profiles') or {}
    instruction = profiles.get('instruction')
    instruction_verified = (
        isinstance(instruction, dict) and instruction.get('status') == 'completed'
        and any(isinstance(artifact, dict) and artifact.get('kind') == 'instruction_listing'
                and artifact.get('path') for artifact in instruction.get('artifacts', [])))
    evidence_categories = {x['category'] for x in hypotheses}
    overlap = 'data_movement' in evidence_categories and bool(
        evidence_categories & {'vector_compute', 'cube_compute'})
    complex_path = kind in {'scan', 'scatter', 'sampling', 'gemm'} or bool(
        case.get('execution_path', {}).get('multiple_kernels'))
    available = valid_count > 0 and not invalid
    needs_instruction = available and (complex_path or overlap or not hypotheses)
    requested, limitations = [], []
    if not available:
        requested.append({'level': 'metrics', 'reason': 'Missing valid pipeline metrics for this candidate'})
    elif needs_instruction and not instruction_verified:
        if 'instruction' in levels:
            requested.append({'level': 'instruction',
                              'reason': 'Complex dependencies or pipeline overlap require instruction, synchronization, and source-region evidence to distinguish mechanisms'})
        else:
            limitations.append('Finer localization is needed, but this service does not declare instruction support; leave the mechanism unresolved')
    if instruction_verified:
        limitations.append('Instruction attachments exist; hotspot regions and causal mechanisms still require analysis')
    elif isinstance(instruction, dict):
        limitations.append('Instruction request is incomplete or lacks a valid instruction_listing; retain device counters/log attachments separately')
    if invalid:
        limitations.append('Metric ratio is outside the valid range or has an invalid type: ' + ', '.join(invalid))
    has_knowledge = (kind in KNOWLEDGE or bool(evidence_categories)
                     or kind == 'copy' or complex_path or overlap
                     or needs_instruction or instruction_verified)
    return {
        'schema_version': '1.0', 'operator': operator, 'operator_type': kind,
        'case_id': case.get('case_id'), 'axes': case.get('axes', {}),
        'stage': 'instruction_evidence' if instruction_verified else 'metrics' if available else 'collect_metrics',
        'hypotheses': hypotheses, 'requested_collections': requested,
        'execution_path_status': 'provided' if case.get('execution_path') else 'needs_reference_comparison',
        'execution_path_action': 'Check candidate/reference kernels, DMA, temporary tensors, and launch counts; ratios alone do not identify the reference path',
        'knowledge': ['operators.md', 'evidence.md'] if has_knowledge else [],
        'instruction_supported': 'instruction' in levels,
        'capabilities': list(capabilities),
        'roofline': {'status': 'not_established', 'reason': 'Physical work and verified hardware limits for the same scope are unavailable'},
        'limitations': limitations,
        'validation': 'Implement suggestions as source changes, pass full correctness, then compare formal performance. Use one full formal evaluation by default; additional runs need an independent reason.',
    }


def enrich(report: dict, service: dict) -> dict:
    profile = service.get('profile', {})
    for case in report['cases']:
        case['hgkda'] = diagnose_case(case, levels=profile.get('levels', []),
                                     capabilities=profile.get('capabilities', []))
    report['diagnostic_policy'] = 'hgkda-staged-v1'
    return report
