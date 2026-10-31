"""按实际计算类型检索知识，避免将所有新目标归为逐元素。"""


def operator_type(name):
    if name in {'native_group_norm','batch_norm_with_update_functional'}:
        return 'normalization'
    if name=='cumulative_trapezoid':
        return 'scan'
    if name=='linalg_inv_ex':
        return 'linear_algebra'
    if name=='uniform':
        return 'random'
    if name=='transform_bias_rescale_qkv':
        return 'layout_fusion'
    if name=='cross_out':
        return 'small_vector'
    if name=='fake_quantize_per_tensor_affine_cachemask_out':
        return 'quantization'
    return 'pointwise'


def reference_documents(name):
    kind=operator_type(name)
    if kind in {'normalization','scan'}:
        return ['算子类型/归约与扫描.md','性能指标/指令与同步.md']
    if kind in {'layout_fusion','small_vector','quantization'}:
        return ['算子类型/融合与执行路径.md','性能指标/搬运与执行路径.md']
    if kind=='linear_algebra':
        return ['算子类型/融合与执行路径.md','性能指标/计算与控制.md']
    if kind=='random':
        return ['算子类型/索引与采样.md','性能指标/计算与控制.md']
    return ['算子类型/逐元素与原地.md','性能指标/搬运与执行路径.md']
