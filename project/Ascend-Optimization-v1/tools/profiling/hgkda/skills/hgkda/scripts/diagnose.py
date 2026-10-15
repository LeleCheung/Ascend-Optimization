#!/usr/bin/env python3
"""按算子类型与硬件流水证据生成分阶段体检计划；不替代正式评测。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

TYPES = {
    'selu_': 'pointwise', 'rad2deg_': 'pointwise', 'log10_': 'pointwise',
    'conj_physical': 'pointwise', 'logaddexp2': 'pointwise',
    'uniform': 'random', 'cumulative_trapezoid': 'scan',
    'amin': 'reduction', 'moe_sum_reduce': 'reduction',
    'per_token_quant_int8': 'reduction',
    'value_selecting_reduction_backward': 'scatter', 'scatter_add': 'scatter',
    'upsample_bilinear2d': 'sampling', 'fractional_max_pool3d': 'sampling',
    'rotary_embedding': 'indexed_pointwise',
    'gelu_and_mul': 'fused_pointwise', 'silu_and_mul_masked': 'fused_pointwise',
    'matmul_bias_activation': 'gemm', 'narrow_copy': 'copy',
}
KNOWLEDGE = {
    'pointwise': '算子类型/逐元素与原地.md',
    'random': '算子类型/归约与扫描.md',
    'scan': '算子类型/归约与扫描.md',
    'reduction': '算子类型/归约与扫描.md',
    'scatter': '算子类型/索引与采样.md',
    'sampling': '算子类型/索引与采样.md',
    'indexed_pointwise': '算子类型/索引与采样.md',
    'fused_pointwise': '算子类型/融合与执行路径.md',
    'gemm': '算子类型/融合与执行路径.md',
    'copy': '算子类型/融合与执行路径.md',
}
METRICS = {
    'aiv_mte2_ratio': ('data_movement', '检查连续性、块大小和重复读入'),
    'aiv_mte3_ratio': ('data_movement', '检查输出搬运、写回粒度和临时结果'),
    'aic_mte2_ratio': ('data_movement', '检查矩阵面板重复搬运、K 分块及计算重叠'),
    'aic_mte1_ratio': ('data_movement', '检查片上矩阵面板搬运与复用'),
    'aiv_vec_ratio': ('vector_compute', '检查超越函数、类型转换和归约指令'),
    'aic_mac_ratio': ('cube_compute', '检查矩阵乘计算效率、tile 与累加路径'),
    'aiv_scalar_ratio': ('control', '检查地址计算、循环和 program 覆盖'),
    'aic_scalar_ratio': ('control', '检查矩阵乘循环、索引和核覆盖'),
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
                               'interpretation': '流水活跃支持优先检查此方向，尚未证明其限制总耗时'})
    profiles = case.get('profiles') or {}
    instruction = profiles.get('instruction')
    instruction_verified = (isinstance(instruction, dict)
                            and bool(instruction.get('artifacts')))
    evidence_categories = {x['category'] for x in hypotheses}
    overlap = 'data_movement' in evidence_categories and bool(
        evidence_categories & {'vector_compute', 'cube_compute'})
    complex_path = kind in {'scan', 'scatter', 'sampling', 'gemm'} or bool(
        case.get('execution_path', {}).get('multiple_kernels'))
    available = valid_count > 0 and not invalid
    needs_instruction = available and (complex_path or overlap or not hypotheses)
    requested, limitations = [], []
    if not available:
        requested.append({'level': 'metrics', 'reason': '缺少有效的同候选流水指标'})
    elif needs_instruction and not instruction_verified:
        if 'instruction' in levels:
            requested.append({'level': 'instruction',
                              'reason': '复杂依赖或流水重叠需要指令、同步与源码热点证据区分'})
        else:
            limitations.append('需要更细定位，但本次服务没有声明 instruction；保留未决机制')
    if instruction_verified:
        limitations.append('指令附件已存在；热点与因果机制仍需分析，不自动把附件存在当作完成诊断')
    if invalid:
        limitations.append('指标超出合法比例范围或类型无效：' + ', '.join(invalid))
    refs = [KNOWLEDGE[kind]] if kind in KNOWLEDGE else []
    if 'data_movement' in evidence_categories or kind == 'copy':
        refs.append('性能指标/搬运与执行路径.md')
    if evidence_categories & {'vector_compute', 'cube_compute', 'control'}:
        refs.append('性能指标/计算与控制.md')
    if complex_path or overlap or needs_instruction or instruction_verified:
        refs.append('性能指标/指令与同步.md')
    return {
        'schema_version': '1.0', 'operator': operator, 'operator_type': kind,
        'case_id': case.get('case_id'), 'axes': case.get('axes', {}),
        'stage': 'instruction_evidence' if instruction_verified else 'metrics' if available else 'collect_metrics',
        'hypotheses': hypotheses, 'requested_collections': requested,
        'execution_path_status': 'provided' if case.get('execution_path') else 'needs_reference_comparison',
        'execution_path_action': '核对候选与参考的 kernel、DMA、临时张量和启动数；不由比例推断参考路径',
        'knowledge': list(dict.fromkeys(refs)),
        'instruction_supported': 'instruction' in levels,
        'capabilities': list(capabilities),
        'roofline': {'status': 'not_established', 'reason': '尚无同范围物理工作量和已验证硬件上限'},
        'limitations': limitations,
        'validation': '每项建议须落到源码变更，完整正确性通过后比较正式性能，最终复验两次',
    }


def enrich(report: dict, service: dict) -> dict:
    profile = service.get('profile', {})
    for case in report['cases']:
        case['hgkda'] = diagnose_case(case, levels=profile.get('levels', []),
                                     capabilities=profile.get('capabilities', []))
    report['diagnostic_policy'] = 'hgkda-staged-v1'
    return report


def knowledge_for(report: dict) -> str:
    root = Path(__file__).resolve().parents[1] / 'references'
    names = list(dict.fromkeys(name for case in report['cases']
                              for name in case.get('hgkda', {}).get('knowledge', [])))
    return '\n\n'.join((root / name).read_text(encoding='utf-8') for name in names)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--service-status', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('输出已存在，请使用新路径以保留原始证据')
    report = enrich(json.loads(a.report.read_text(encoding='utf-8-sig')),
                    json.loads(a.service_status.read_text(encoding='utf-8-sig')))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'cases': len(report['cases']), 'output': str(a.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
