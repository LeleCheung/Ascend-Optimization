# CANN统一参考下的有效交付进展

2026-11-17更新。主图是有效候选进展摘要；原始事件、失败与探索退步仍保留。原始送测FlagGems未通过时，允许起点画叉，后续无效／待审查候选不在主图画叉。

## 换算方法

`重算加速比 = exp(mean(log(交付评测中的逐case CANN参考耗时 / 该历史版本逐case候选耗时)))`。

固定当前交付测量的CANN参考耗时。核对完整性能用例配置、seed、用例集合、预热时间、计时时间与试验次数后重算，不用两个总加速比相除，也不将缺失值补成0。_conj_physical仅归一化名称前缀。

**≈表示重算值，交付终点为CANN实测值。** 重算消除不同参考分母造成的跳变，但不消除历史设备负载、时钟、环境或计时波动；不是新的真机实验，也不证明跨时刻因果。不同正确性容差与超时设置仍记录在原始请求中，计时设置匹配不代表所有正确性合同完全一致。

## 主图筛选

- 失败或已知不合规候选移入详情；静态筛查发现待审查入口但缺少对应完整审查的候选，也先不进主图，不能直接称其违规。
- 无profiling／有profiling对照组、重复基线、独立重生成，属于另一条线，不能作为本线连续迭代。
- 有效但未改善显示进展的探索点移入分支，并明确标为有效退步。该摘要是事后展示规则，不冒充原控制器逐轮保留决策。
- 固定交付终点始终保留，哪怕比某个历史重算点低；不改成绩、不人为压低原生基线、不保证曲线单调。
- 保留源码继承证明；筛选和换算不会自动补齐父版本关系。

| 算子 | 主图点数 | 起点 | 交付CANN实测 |
| --- | ---: | --- | ---: |
| [selu_](selu_/index.html) | 5 | ≈0.147× | 1.039× |
| [rad2deg_](rad2deg_/index.html) | 4 | 原生未通过 | 0.834× |
| [log10_](log10_/index.html) | 8 | 原生未通过 | 0.941× |
| [uniform](uniform/index.html) | 3 | ≈0.218× | 2.399× |
| [value_selecting_reduction_backward](value_selecting_reduction_backward/index.html) | 2 | ≈0.706× | 1.349× |
| [scatter_add](scatter_add/index.html) | 4 | 原生未通过 | 1.892× |
| [upsample_bilinear2d](upsample_bilinear2d/index.html) | 4 | 原生未通过 | 1.142× |
| [_conj_physical](_conj_physical/index.html) | 4 | 原生未通过 | 1.234× |
| [negative_](negative_/index.html) | 3 | 原生未通过 | 0.806× |
| [xlogy_scalar_tensor](xlogy_scalar_tensor/index.html) | 2 | ≈0.206× | 1.536× |
| [nan_to_num_](nan_to_num_/index.html) | 3 | 原生未通过 | 0.863× |
| [cross_out](cross_out/index.html) | 2 | ≈0.149× | 3.868× |
| [hardshrink_backward](hardshrink_backward/index.html) | 3 | ≈0.151× | 0.999× |
| [subtract_](subtract_/index.html) | 2 | 原生未通过 | 1.659× |
| [native_group_norm](native_group_norm/index.html) | 2 | 原生未通过 | 1.004× |
| [log2_](log2_/index.html) | 3 | 原生未通过 | 0.880× |
| [transform_bias_rescale_qkv](transform_bias_rescale_qkv/index.html) | 6 | ≈0.003× | 0.941× |
| [fix](fix/index.html) | 4 | 原生未通过 | 0.969× |
| [round_out](round_out/index.html) | 6 | 原生未通过 | 0.916× |
| [hardswish_out](hardswish_out/index.html) | 5 | ≈0.143× | 0.867× |
| [fake_quantize_per_tensor_affine_cachemask_out](fake_quantize_per_tensor_affine_cachemask_out/index.html) | 2 | ≈6.894× | 8.155× |
| [special_round_out](special_round_out/index.html) | 8 | 原生未通过 | 1.000× |
| [special_xlogy_](special_xlogy_/index.html) | 3 | 原生未通过 | 1.071× |
| [hardswish](hardswish/index.html) | 4 | ≈0.168× | 0.849× |
| [fmod_tensor_](fmod_tensor_/index.html) | 3 | 原生未通过 | 0.841× |
| [batch_norm_with_update_functional](batch_norm_with_update_functional/index.html) | 3 | ≈1.813× | 1.774× |
