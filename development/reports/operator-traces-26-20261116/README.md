# 26项优化轨迹图册

[打开图册](index.html) · [完整数据索引](catalog.json) · [版式及链接检查](presentation-check.json)

22项有多个展示节点，4项以交付结果卡展示。hardswish_out已从回归／修复目录恢复同合同两点；log2_保留早期单点在数据中，图面展示当前交付结果。

每图选择同一原始账本的一条主要尝试线；log10使用已恢复的连续主线。账本尝试顺序不等于逐次源码直接继承。不同参考、合同、起点和批次不拼接。每项HTML展开区及trace.json保存全部已收集历史与原计划；主线节点少不代表该算子总共只优化了这些轮次。

箭头颜色表示记录中的判断依据类别，节点形状表示修改位置。来源不明保留待核验；历史归类、模型声明和工具共现不能证明独立因果。缺失轨迹及历史参考限制保留在对应页面。

| 算子 | 主线节点 | 已收集历史记录 | 范围 |
| --- | ---: | ---: | --- |
| [selu_](selu_/index.html) | 3 | 10 | 已保留选线与历史来源 |
| [rad2deg_](rad2deg_/index.html) | 4 | 8 | 已保留选线与历史来源 |
| [log10_](log10_/index.html) | 12 | 9 | 已保留选线与历史来源 |
| [uniform](uniform/index.html) | 3 | 11 | 已保留选线与历史来源 |
| [value_selecting_reduction_backward](value_selecting_reduction_backward/index.html) | 2 | 6 | 已保留选线与历史来源 |
| [scatter_add](scatter_add/index.html) | 3 | 10 | 已保留选线与历史来源 |
| [upsample_bilinear2d](upsample_bilinear2d/index.html) | 4 | 8 | 已保留选线与历史来源 |
| [_conj_physical](_conj_physical/index.html) | 4 | 4 | 已保留选线与历史来源 |
| [negative_](negative_/index.html) | 3 | 3 | 已保留选线与历史来源 |
| [xlogy_scalar_tensor](xlogy_scalar_tensor/index.html) | 2 | 2 | 已保留选线与历史来源 |
| [nan_to_num_](nan_to_num_/index.html) | 5 | 8 | 已保留选线与历史来源 |
| [cross_out](cross_out/index.html) | 5 | 6 | 已保留选线与历史来源 |
| [hardshrink_backward](hardshrink_backward/index.html) | 4 | 6 | 已保留选线与历史来源 |
| [subtract_](subtract_/index.html) | 2 | 2 | 已保留选线与历史来源 |
| [native_group_norm](native_group_norm/index.html) | 3 | 4 | 已保留选线与历史来源 |
| [log2_](log2_/index.html) | 1 | 1 | 仅找到1次正式测量，优化过程记录待补 |
| [transform_bias_rescale_qkv](transform_bias_rescale_qkv/index.html) | 1 | 0 | 已保存交付结果；原优化轮次记录待补 |
| [fix](fix/index.html) | 4 | 6 | 已保留选线与历史来源 |
| [round_out](round_out/index.html) | 4 | 5 | 已保留选线与历史来源 |
| [hardswish_out](hardswish_out/index.html) | 2 | 0 | 已保留选线与历史来源 |
| [fake_quantize_per_tensor_affine_cachemask_out](fake_quantize_per_tensor_affine_cachemask_out/index.html) | 1 | 0 | 已保存交付结果；原优化轮次记录待补 |
| [special_round_out](special_round_out/index.html) | 4 | 6 | 已保留选线与历史来源 |
| [special_xlogy_](special_xlogy_/index.html) | 3 | 3 | 已保留选线与历史来源 |
| [hardswish](hardswish/index.html) | 4 | 6 | 已保留选线与历史来源 |
| [fmod_tensor_](fmod_tensor_/index.html) | 2 | 2 | 已保留选线与历史来源 |
| [batch_norm_with_update_functional](batch_norm_with_update_functional/index.html) | 1 | 0 | 已保存交付结果；原优化轮次记录待补 |

## 重建

```powershell
python -m development.tools.traces.catalog --output development/reports/operator-traces-26-20261116 --png
```

SVG／JSON仅需Python标准库；PNG及版式检查需要可选Playwright和Edge。图例缩小为24像素置于右上角，仅显示本图使用的类别；原始参考口径与记录数量保留在详情及JSON。Profiling合并展示成对采集、宿主剖析和运行探测，原始分类不改写。方法标注不相互重叠，曲线不用阶梯转角。
