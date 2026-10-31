"""清点本机CANN头文件与导出符号；不把接口存在当作已合格性能基线。"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path

MAPPING = [
    ('selu_', 'aclnn_selu.h', 'aclnnInplaceSelu', '需核验原地语义'),
    ('rad2deg_', 'aclnn_mul.h', 'aclnnInplaceMuls', '乘常量组合；不是专用rad2deg接口'),
    ('log10_', 'aclnn_log10.h', 'aclnnInplaceLog10', '需核验特殊值与原地语义'),
    ('cumulative_trapezoid', 'aclnn_cumsum.h', 'aclnnCumsum', '仅扫描部分；还需相邻加法、缩放及dtype舍入'),
    ('uniform', 'aclnn_uniform.h', 'aclnnInplaceUniform', '需核验随机算法、seed、offset与状态推进'),
    ('value_selecting_reduction_backward', None, None, '需由scatter类接口组合并验证布局'),
    ('scatter_add', 'aclnn_scatter_add.h', 'aclnnScatterAdd', '需核验输入重复索引及dtype'),
    ('upsample_bilinear2d', 'aclnn_upsample_bilinear_2d.h', 'aclnnUpsampleBilinear2d', '需核验对齐、尺度和布局'),
    ('fractional_max_pool3d', None, None, '本清单未指定专用aclnn接口；独立NPU分解对照另列'),
    ('_conj_physical', None, None, '本清单未指定专用aclnn接口；共轭、输出别名和布局需单独验证'),
    ('negative_', 'aclnn_neg.h', 'aclnnInplaceNeg', '需核验原地语义'),
    ('xlogy_scalar_tensor', None, None, '未指定专用接口；组合log与乘法需特殊值检查'),
    ('nan_to_num_', 'aclnn_nan_to_num.h', 'aclnnInplaceNanToNum', '需核验NaN、无穷与默认替换值'),
    ('cross_out', 'aclnn_linalg_cross.h', 'aclnnLinalgCross', '需核验dim、广播与out别名'),
    ('expm1_', 'aclnn_expm1.h', 'aclnnInplaceExpm1', '本轮FP16/FP32直接；BF16兼容组合；扩展精度失败保留'),
    ('hardshrink_backward', 'aclnn_hardshrink_backward.h', 'aclnnHardshrinkBackward', '需核验阈值边界'),
    ('subtract_', 'aclnn_sub.h', 'aclnnInplaceSub', '需核验alpha、广播与别名'),
    ('native_group_norm', 'aclnn_group_norm.h', 'aclnnGroupNorm', '需核验多输出、统计量与eps'),
    ('log2_', 'aclnn_log2.h', 'aclnnInplaceLog2', '需核验特殊值与原地语义'),
    ('transform_bias_rescale_qkv', 'aclnn_transform_bias_rescale_qkv.h', 'aclnnTransformBiasRescaleQkv', '需核验布局与多输出'),
    ('fix', 'aclnn_trunc.h', 'aclnnTrunc', '本轮已按原合同及补充检查核验'),
    ('round_out', 'aclnn_round.h', 'aclnnRound', '需核验decimals、舍入模式与out布局'),
    ('hardswish_out', 'aclnn_hardswish.h', 'aclnnHardswish', '需核验out别名及布局'),
    ('fake_quantize_per_tensor_affine_cachemask_out', 'aclnn_fake_quant_per_tensor_affine_cachemask.h', 'aclnnFakeQuantPerTensorAffineCachemask', '本轮采用FP32兼容组合；量化参数和out语义需核验'),
    ('special_round_out', 'aclnn_round.h', 'aclnnRound', '需核验decimals与out布局'),
    ('special_xlogy_', None, None, '未指定专用接口；需特殊值与原地语义检查'),
    ('hardswish', 'aclnn_hardswish.h', 'aclnnHardswish', '需核验dtype及布局'),
    ('fmod_tensor_', 'aclnn_fmod_tensor.h', 'aclnnInplaceFmodTensor', '需核验符号、零除数与广播'),
    ('linalg_inv_ex', 'aclnn_inverse.h', 'aclnnInverse', '只提供inverse，不提供info；不是完整inv_ex接口'),
    ('batch_norm_with_update_functional', 'aclnn_batch_norm.h', 'aclnnBatchNorm', '本轮FP32兼容组合；状态不变及六输出语义需核验'),
]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/usr/local/Ascend/cann-9.0.0'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    headers = {}
    for path in a.root.rglob('aclnn_*.h'):
        if path.is_file(): headers.setdefault(path.name, []).append(path)
    lib = ctypes.CDLL('libopapi.so')
    rows = []
    for op, name, symbol, note in MAPPING:
        paths = sorted(headers.get(name, []))
        exports = {s: bool(getattr(lib, s, None)) for s in [symbol, symbol+'GetWorkspaceSize']} if symbol else {}
        rows.append(dict(operator=op, header_name=name, expected_symbol=symbol, exports=exports,
            headers=[dict(path=str(path), sha256=sha(path), declarations=[line.strip() for line in path.read_text(errors='replace').splitlines() if symbol and symbol in line]) for path in paths],
            status='header_and_exports_present' if paths and exports and all(exports.values()) else 'requires_mapping_or_verification',
            semantic_note=note, qualification='not_inferred_from_inventory'))
    result = dict(schema='hgkda-cann-api-inventory-v1', root=str(a.root), operators=rows,
        script_source=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__))),
        scope='只读头文件及动态库符号；未运行设备、不证明dtype或语义支持、不产生性能成绩。缺失仅表示本清单映射未证实。')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
    print(json.dumps(dict(total=len(rows),header_and_exports_present=sum(r['status']=='header_and_exports_present' for r in rows)),ensure_ascii=False))

if __name__ == '__main__': main()
