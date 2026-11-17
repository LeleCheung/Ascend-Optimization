# 26项：FlagGems原始起点—历史版本—交付终点

日期：2026-11-17。只读查档，未补跑设备实验。

固定FlagGems：`d6a8eec473517a3d68157b208eb9c057eb1d4c50`。26项的原始源文件均与当时计划记录SHA一致，26份起点送测源码的原始模块AST均未改变，末尾仅追加评测入口。该核验不自动证明动态后端派发路径。

**已找到全部26项的基线送测记录和交付源码／结果。** 原始FlagGems送测基线与优化账本中首次修改后的baseline是两类记录，旧图只读取后者。

下表起点分数为当时参考下的历史记录，终点为当前CANN参考；两列不能直接相除解释为优化收益。失败保留原状态。中间测量数量不等于模型轮次，包含不同批次及重复核验；厂商参考测量单独标记。

| 算子 | 原生送测起点 | 候选测量／源码数 | 交付 CANN | 完整评测 | 证据 |
| --- | --- | ---: | ---: | ---: | --- |
| `selu_` | 0.119× | 30／17 | 1.039× | 36/36 | [selu_](selu_.md) |
| `rad2deg_` | PARTIAL_PASS | 16／8 | 0.834× | 36/36 | [rad2deg_](rad2deg_.md) |
| `log10_` | PARTIAL_PASS | 44／32 | 0.941× | 36/36 | [log10_](log10_.md) |
| `uniform` | 0.141× | 9／4 | 2.399× | 18/18 | [uniform](uniform.md) |
| `value_selecting_reduction_backward` | 0.759× | 21／7 | 1.349× | 96/96 | [value_selecting_reduction_backward](value_selecting_reduction_backward.md) |
| `scatter_add` | RUNTIME_ERROR | 14／7 | 1.892× | 18/18 | [scatter_add](scatter_add.md) |
| `upsample_bilinear2d` | RUNTIME_ERROR | 19／9 | 1.142× | 24/24 | [upsample_bilinear2d](upsample_bilinear2d.md) |
| `_conj_physical` | 1.723× | 12／6 | 1.234× | 14/14 | [_conj_physical](_conj_physical.md) |
| `negative_` | PARTIAL_PASS | 17／12 | 0.806× | 30/30 | [negative_](negative_.md) |
| `xlogy_scalar_tensor` | 0.148× | 7／3 | 1.536× | 30/30 | [xlogy_scalar_tensor](xlogy_scalar_tensor.md) |
| `nan_to_num_` | PARTIAL_PASS | 12／7 | 0.863× | 30/30 | [nan_to_num_](nan_to_num_.md) |
| `cross_out` | 0.136× | 15／9 | 3.868× | 21/21 | [cross_out](cross_out.md) |
| `hardshrink_backward` | 0.154× | 14／8 | 0.999× | 30/30 | [hardshrink_backward](hardshrink_backward.md) |
| `subtract_` | PARTIAL_PASS | 6／2 | 1.659× | 30/30 | [subtract_](subtract_.md) |
| `native_group_norm` | PARTIAL_PASS | 13／7 | 1.004× | 27/27 | [native_group_norm](native_group_norm.md) |
| `log2_` | PARTIAL_PASS | 14／6 | 0.880× | 30/30 | [log2_](log2_.md) |
| `transform_bias_rescale_qkv` | 0.004× | 11／5 | 0.941× | 21/21 | [transform_bias_rescale_qkv](transform_bias_rescale_qkv.md) |
| `fix` | RUNTIME_ERROR | 26／15 | 0.969× | 30/30 | [fix](fix.md) |
| `round_out` | PARTIAL_PASS | 19／10 | 0.916× | 30/30 | [round_out](round_out.md) |
| `hardswish_out` | 0.177× | 21／12 | 0.867× | 30/30 | [hardswish_out](hardswish_out.md) |
| `fake_quantize_per_tensor_affine_cachemask_out` | 8.651× | 5／2 | 8.155× | 30/30 | [fake_quantize_per_tensor_affine_cachemask_out](fake_quantize_per_tensor_affine_cachemask_out.md) |
| `special_round_out` | RUNTIME_ERROR | 22／12 | 1.000× | 30/30 | [special_round_out](special_round_out.md) |
| `special_xlogy_` | PARTIAL_PASS | 6／2 | 1.071× | 30/30 | [special_xlogy_](special_xlogy_.md) |
| `hardswish` | 0.148× | 21／12 | 0.849× | 30/30 | [hardswish](hardswish.md) |
| `fmod_tensor_` | RUNTIME_ERROR | 16／9 | 0.841× | 30/30 | [fmod_tensor_](fmod_tensor_.md) |
| `batch_norm_with_update_functional` | 100.912× | 5／2 | 1.774× | 27/27 | [batch_norm_with_update_functional](batch_norm_with_update_functional.md) |

## 本轮查清的事实

- 原生送测源码、优化器第一轮和后续重生成起点必须分别记录。26项原生送测记录都已找到，不能再以优化账本第一点代替它。10月21日批次覆盖早期8项交付算子，10月29日批次覆盖其余18项；按固定批次选起点，没有挑选最低分。
- 原来单点的QKV、log2、hardswish_out已找到多份中间候选；cachemask与batch_norm也找到基线、参考切换和严格Triton恢复记录。
- 并非26项都从很低的有效成绩开始。cachemask原历史参考下为8.651×、batch_norm为100.912×，均需结合参考执行路径解释，不与CANN交付分数直接比较。
- 原生失败不统一归为算子错误：scatter_add早期记录是相对导入失败，上采样早期记录包含参考接口错误；后续修正基线保留在各项完整清单。
- 本轮只完成证据清单，不声称497条测量已经形成26条无缺口的继承链。下一步按源码SHA和父版本记录接线；不同参考的段可同图展示，不能把参考切换当作性能收益。

## 范围与复现

检索两个服务器历史根目录，找到497份测量、43份基线目录测量，检索解析错误0项。所有导出的源码／基线原件SHA已核验，26份交付源码和结果与当前索引一致。

源码与基线原件：[原始检索索引](../../experiments/lineage/20261117/source-evidence/inventory.json)；完整结构化清单：[inventory.json](inventory.json)。

```powershell
python -m development.tools.traces.build_lineage_inventory --evidence development/experiments/lineage/20261117/source-evidence --output development/reports/native-to-delivery-26-20261117
```
