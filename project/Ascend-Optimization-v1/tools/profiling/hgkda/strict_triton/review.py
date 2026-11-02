"""登记本轮人工源码审查结论；必须先阅读实际源码，非自动静态证明。"""
from pathlib import Path
from common import BATCH,HERE,PROJECT,read,save,sha
from gate import inspect

NOTES={
    'selu_':'连续/对齐/分层stride路径均调用Triton；缓存不保存Tensor或输出',
    'rad2deg_':'截断Ascend C桥后恢复run，连续与stride两个kernel及当前stream启动',
    'log10_':'仅_triton_run；nd/scaled/streaming三种设备路径保留，外部so加载全部移除',
    'cumulative_trapezoid':'扫描前转置和扫描后还原均Triton；默认参考不稳定仍单列',
    'uniform':'Philox计算、dtype区间夹取由Triton完成；struct只计算宿主标量边界，生成器仅管理状态',
    'value_selecting_reduction_backward':'写零和索引梯度回填均Triton，非连续路径也是Triton',
    'scatter_add':'复制输入、索引排序、累加及布局路径均Triton；cann.extension为Triton编译器DSL扩展',
    'upsample_bilinear2d':'三种插值路径均Triton；输出torch.empty只分配',
    'fractional_max_pool3d':'Triton池化，样本连续化和可选随机采样改成内嵌Triton；原_random_samples路径与合同不变',
    '_conj_physical':'复制和共轭均Triton；conj位和view_as_real是元数据视图，不调用物理共轭目标接口',
    'negative_':'移除Ascend C桥，原Triton符号位翻转、连续和stride路径恢复',
    'xlogy_scalar_tensor':'flat/nd/scaled路径Triton，标量self只作运行时参数',
    'nan_to_num_':'所有默认/动态参数和布局路径Triton；按shape/地址缓存编译对象，不能缓存结果',
    'cross_out':'out别名输入的保护副本改成Triton，叉积flat/general仍Triton',
    'expm1_':'移除Ascend C桥，恢复Triton小值多项式及指数路径，精度仍须原补测',
    'hardshrink_backward':'输入连续化及grad_input回写改成Triton；原Triton主计算恢复',
    'subtract_':'连续与独立stride广播路径均Triton，无厂商算子调用',
    'native_group_norm':'输入和独立weight/bias连续化改成Triton，主归约与输出仍Triton',
    'log2_':'移除外部设备库，恢复原Triton flat/strided路径',
    'transform_bias_rescale_qkv':'首轮直接恢复原Triton实现，无外部宿主桥加载，Q/K/V共享分配但区域不重叠',
    'fix':'仅原Triton核；_empty_like只是torch.empty分配别名，数值转换在Triton',
    'round_out':'移除Ascend C桥，回退输入连续化等搬运改成Triton，decimals仍按原语义检查',
    'hardswish_out':'提取原Triton out计算，连续化和独立out回写均改成Triton',
    'fake_quantize_per_tensor_affine_cachemask_out':'输入连续化、out及mask回写改成Triton，量化仍原Triton公式',
    'special_round_out':'移除Ascend C桥；原输入/输出stride和运行时decimals路径保留',
    'special_xlogy_':'flat与广播stride路径均Triton',
    'hardswish':'移除Ascend C桥，连续化改成Triton，主Triton流式核保留',
    'fmod_tensor_':'移除Ascend C桥，原Triton fmod计算及特殊值路径保留',
    'linalg_inv_ex':'矩阵预处理、逆矩阵转换均Triton；check_errors分支仅在宿主检查info后抛错，不在CPU求逆',
    'batch_norm_with_update_functional':'flatten潜在复制及默认运行统计填零/一改成Triton；归约/归一化及状态更新原Triton',
}

def main():
    index=read(BATCH/'candidate-index.json')
    dependency_paths=['utils/libentry.py','utils/random_utils.py','utils/tl_extra_shim.py','runtime/__init__.py']
    deps={}
    for name in dependency_paths:
        p=PROJECT.parents[1]/'third_party/FlagGems/src/flag_gems'/name
        if not p.exists():p=PROJECT.parent.parent/'third_party/FlagGems/src/flag_gems'/name
        if not p.exists():continue
        out=BATCH/'dependencies/flag_gems'/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(p.read_bytes())
        deps[str(p)]=dict(sha256=sha(p),snapshot=str(out))
    assert len(deps)>=3,'固定FlagGems依赖缺失'
    save(BATCH/'dependency-review.json',dict(status='reviewed',files=deps,
        notes='libentry仅负责Triton编译/缓存/启动；random_utils均匀转换为Triton JIT；runtime设备上下文仅管设备；外部算子函数导入不进入候选'))
    for op,v in index.items():
        code=Path(v['source']);audit=inspect(code);assert audit['static_passed'],audit
        v.update(status='reviewed_strict_triton',all_branches_reviewed=True,dependencies_reviewed=True,
            review_note=NOTES[op],reviewer='Codex逐文件审查原源码、移除的设备库、保留kernel与全部宿主调用',
            dependency_review_sha256=sha(BATCH/'dependency-review.json'))
        save(code.parent/'review.json',v)
    save(BATCH/'candidate-index.json',index)
    print('REVIEWED',len(index))

if __name__=='__main__':main()
