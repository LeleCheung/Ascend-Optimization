# 26项源码接续轨迹

每图从原始FlagGems送测记录出发，以当前交付测量结束。实线仅表示同源码／格式或入口适配等价，或显式父版本记录；虚线保留尚未核实的接续。换参考的连接单独标注，不称为性能提升。

连续编号是展示索引，不冒充完整模型轮数；主线优先选择可证的交付祖先，其余尝试保留在全部历史中。

| 算子 | 主线点 | 根到交付来历 | 同源码两侧参考 | 交付源码两侧参考 |
| --- | ---: | --- | --- | --- |
| [selu_](selu_/index.html) | 28 | 部分待接 | 有 | 有 |
| [rad2deg_](rad2deg_/index.html) | 6 | 已找到 | 有 | 有 |
| [log10_](log10_/index.html) | 10 | 已找到 | 有 | 有 |
| [uniform](uniform/index.html) | 7 | 已找到 | 有 | 有 |
| [value_selecting_reduction_backward](value_selecting_reduction_backward/index.html) | 5 | 已找到 | 有 | 有 |
| [scatter_add](scatter_add/index.html) | 5 | 已找到 | 有 | 有 |
| [upsample_bilinear2d](upsample_bilinear2d/index.html) | 5 | 已找到 | 有 | 有 |
| [_conj_physical](_conj_physical/index.html) | 6 | 已找到 | 有 | 有 |
| [negative_](negative_/index.html) | 5 | 已找到 | 有 | 最新交付仅CANN |
| [xlogy_scalar_tensor](xlogy_scalar_tensor/index.html) | 4 | 已找到 | 有 | 有 |
| [nan_to_num_](nan_to_num_/index.html) | 4 | 已找到 | 有 | 有 |
| [cross_out](cross_out/index.html) | 6 | 已找到 | 有 | 有 |
| [hardshrink_backward](hardshrink_backward/index.html) | 12 | 已找到 | 有 | 有 |
| [subtract_](subtract_/index.html) | 4 | 已找到 | 有 | 有 |
| [native_group_norm](native_group_norm/index.html) | 4 | 已找到 | 有 | 有 |
| [log2_](log2_/index.html) | 8 | 部分待接 | 有 | 有 |
| [transform_bias_rescale_qkv](transform_bias_rescale_qkv/index.html) | 8 | 已找到 | 有 | 有 |
| [fix](fix/index.html) | 6 | 已找到 | 有 | 有 |
| [round_out](round_out/index.html) | 9 | 已找到 | 有 | 最新交付仅CANN |
| [hardswish_out](hardswish_out/index.html) | 11 | 部分待接 | 有 | 最新交付仅CANN |
| [fake_quantize_per_tensor_affine_cachemask_out](fake_quantize_per_tensor_affine_cachemask_out/index.html) | 3 | 已找到 | 有 | 有 |
| [special_round_out](special_round_out/index.html) | 19 | 部分待接 | 有 | 最新交付仅CANN |
| [special_xlogy_](special_xlogy_/index.html) | 4 | 已找到 | 有 | 有 |
| [hardswish](hardswish/index.html) | 8 | 已找到 | 有 | 有 |
| [fmod_tensor_](fmod_tensor_/index.html) | 6 | 已找到 | 有 | 有 |
| [batch_norm_with_update_functional](batch_norm_with_update_functional/index.html) | 3 | 已找到 | 有 | 有 |

## 参考切换影响多大

26项都找到至少一个同源码的PyTorch／CANN双参考版本，其中22项就是当前交付源码。四项最新交付目前只找到CANN结果：`negative_`、`round_out`、`hardswish_out`、`special_round_out`。这四项的较早版本仍有双参考记录。

| 同一份源码 | PyTorch参考 | CANN参考 |
| --- | ---: | ---: |
| `upsample_bilinear2d` | 0.915027× | 1.146535× |
| `transform_bias_rescale_qkv` | 1.042504× | 0.931237× |
| `log2_` | 0.634218× | 0.845795× |
| `_conj_physical` | 45.279184× | 1.152554× |
| `batch_norm_with_update_functional` | 23.562510× | 1.748443× |

以上是相同源码的不同参考测量，包含各次测量波动；不是换代码产生的优化增益。源码接续可以成立，性能解释必须分段。历史PyTorch含CPU回退的已知限制继续保留，尤其不能把_conj_physical的大比值当作原生CANN收益。

## 距离完整交付路线

26项原生送测与交付终点均定位；本轮按显式父版本、批次输入到选中输出以及静态等价源码，找到6项根到交付的主要版本来历：`rad2deg_`、`log10_`、`uniform`、`value_selecting_reduction_backward`、`scatter_add`、`upsample_bilinear2d`、`_conj_physical`、`negative_`、`xlogy_scalar_tensor`、`nan_to_num_`、`cross_out`、`hardshrink_backward`、`subtract_`、`native_group_norm`、`transform_bias_rescale_qkv`、`fix`、`round_out`、`fake_quantize_per_tensor_affine_cachemask_out`、`special_xlogy_`、`hardswish`、`fmod_tensor_`、`batch_norm_with_update_functional`。这不表示每次内部源码编辑都有独立性能点。

其余20项已有多个可连接片段，但至少一处来源接续尚未核实。图中虚线明确标注，完整JSON保留缺口；不因参考切换而把可证的同源码接续也判成缺失。

起点真实性也限制“26项都从很低性能开始”的表述：固定批次的26项原始起点中，9项正确性通过且历史加速比低于0.8，14项未完整通过，3项原历史参考下已不低于0.8。原生失败不能填为0分，原生已较快的不能改成低分。

当前图的主线折叠同源码同参考重复测量，但保留账本首轮及最终交付；有完整来历时优先展示交付祖先。其余尝试可展开查看，已知Ascend C混合历史分支保留在此前证据清单，不混入严格Triton主线。未知合规性的历史点不自动认证为严格Triton。

实际提交的父子版本记录位于`development/experiments/lineage/20261117/links-v2/`。补取426份来源文档、210个合同包身份和241份哈希匹配源码；19个引用源文件未在所查路径按原SHA找到，明确记录在links.json。没有新跑设备实验。

## 下一步

先补20项的父版本引用／原始调用连接；不需要把26项全部重做。只有缺少所需参考下测量、或希望把所有点重画到同一性能标尺，才安排统一复测。历史重新测量应单列，不能冒充当年过程。

```powershell
python -m development.tools.traces.stitched --output development/reports/operator-traces-stitched-20261117 --png
```

