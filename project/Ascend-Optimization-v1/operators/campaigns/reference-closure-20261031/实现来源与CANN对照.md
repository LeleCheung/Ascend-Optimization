# 实现来源与CANN对照

## 是否都是手写

不是。当前原参考稳定达标24项中，15项交付后端为Triton，9项为自写Ascend C＋Triton特殊布局回退。后端不等于作者归属：Triton候选包含KG＋CC/DeepSeek生成、HG-KDA辅助生成和Codex修改；近期9项Ascend C候选含Codex直接编写。每项实际来源以冻结源码、生成轨迹和交付清单为准，不能统称全自动Agent优化。

## CANN已有实现为什么还需核验

CANN的aclnn接口可以直接作为厂商参考；Ascend C用于编写自定义算子，二者不是同一份实现。首先对齐输入、dtype、原地／out别名、多输出和stream，再在原KGS输入及容差下比较。PyTorch是否真正调用这个接口，需要设备trace或源码证据，不能仅凭函数名推定。

本机CANN 9.0.0接口清单覆盖30项目标，25项找到了相关头文件和导出符号；其中有的只是组成步骤。该数量不是25项已通过测试。完整记录见[接口清单](research/cann-api-inventory.json)。

|对象|本轮厂商参考|关键区别|
|---|---|---|
|fix|aclnnTrunc|已完成原合同与补充检查；候选与厂商同合同对照继续排队|
|expm1_|aclnnInplaceExpm1／aclnnExpm1|BF16需显式兼容路径；极小数值补测存在失败|
|fake_quantize_per_tensor_affine_cachemask_out|aclnnFakeQuantPerTensorAffineCachemask|本轮以FP32兼容组合覆盖低精度；不是低精度单kernel|
|batch_norm_with_update_functional|aclnnBatchNorm|需复制运行状态、适配形状与统计输出；组合整体成本计入|
|linalg_inv_ex|aclnnInverse|只有inverse，没有info；奇异输入逐case隔离核验，不视为完整inv_ex参考|
|cumulative_trapezoid|aclnnCumsum|只是扫描步骤，仍需增量计算和精度／累加顺序处理|
|fractional_max_pool3d、_conj_physical|本轮未核验专用单算子|历史NPU分解作为独立命名补充参考；不冒充官方单kernel|

## 本轮求逆的新证据

固定PyTorch版本源码显示：连续实数矩阵采用转置LU分解，随后使用adjoint三角求解。单CPU线程实验对四份正确性输入复现了该路径，输出与同版本CPU inv_ex逐位一致；记录位于[CPU中间值](analysis/linalg_inv_ex/cpu-lapack-intermediates.json)，固定源码出处见[provenance](research/pytorch-inverse/provenance.json)。

重复行矩阵在精确数学上不可逆，但转置LU的有限精度余量会使本机参考返回info=0和巨大有限数；原直接LU候选检测到零主元。这解释了原5/7中两项失败的机制背景，不表示已经有新的合格NPU候选。通用最小分块提案、厂商求逆和完整KGS仍按实际结果登记，不缓存或硬编码参考输出。

原合同稳定达标、确定性条件、CPU回退参考及新增CANN／NPU对照分别统计，进度见[30项验收总表](30项验收总表.md)。
