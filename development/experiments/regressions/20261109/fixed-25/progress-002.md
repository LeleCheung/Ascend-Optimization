# 独立 HG-KDA 固定候选回归

状态：running；已处理 11/25；本轮达标 10 项。

候选源码、原输入、容差和计时参数固定；本轮每项一次完整 KGS 正式评测，原补测独立执行。

加速比为完整调用的几何平均；CANN 参考类型含原生单接口及已审核的同语义组合，逐项保存在 JSON 的 reference_kind 字段中。

| 算子 | 拆分前相对 CANN | 本轮相对 CANN | 完整评测 | PyTorch 语义补测 | CANN 语义补测 | 本轮状态 |
| --- | ---: | ---: | --- | --- | --- | --- |
| `selu_` | 0.982× | 1.039× | 36/36 | 42/42 passed | 42/42 passed | 达标 |
| `rad2deg_` | 0.922× | 0.834× | 36/36 | 42/42 passed | 42/42 passed | 达标 |
| `log10_` | 0.843× | 0.941× | 36/36 | 66/66 passed | 66/66 passed | 达标 |
| `value_selecting_reduction_backward` | 1.470× | 1.349× | 96/96 | 96/96 passed | 96/96 passed | 达标 |
| `scatter_add` | 1.886× | 1.892× | 18/18 | 21/21 passed | 21/21 passed | 达标 |
| `upsample_bilinear2d` | 1.147× | 1.142× | 24/24 | 30/30 passed | 30/30 passed | 达标 |
| `_conj_physical` | 1.153× | 1.234× | 14/14 | 16/16 passed | 16/16 passed | 达标 |
| `negative_` | 0.814× | 0.653× | 30/30 | 63/63 passed | 63/63 passed | 相对 CANN 未超过 0.8× |
| `xlogy_scalar_tensor` | 1.579× | 1.536× | 30/30 | 72/72 passed | 72/72 passed | 达标 |
| `nan_to_num_` | 1.100× | 0.863× | 30/30 | 54/54 passed | 54/54 passed | 达标 |
| `cross_out` | 4.090× | 3.868× | 21/21 | 45/45 passed | 45/45 passed | 达标 |
| `hardshrink_backward` | 1.042× | 0.999× | 30/30 | — | — | running |
| `subtract_` | 1.556× | — | — | — | — | pending |
| `native_group_norm` | 0.987× | — | — | — | — | pending |
| `log2_` | 0.846× | — | — | — | — | pending |
| `transform_bias_rescale_qkv` | 0.931× | — | — | — | — | pending |
| `fix` | 1.014× | — | — | — | — | pending |
| `round_out` | 0.829× | — | — | — | — | pending |
| `hardswish_out` | 0.862× | — | — | — | — | pending |
| `fake_quantize_per_tensor_affine_cachemask_out` | 7.060× | — | — | — | — | pending |
| `special_round_out` | 0.946× | — | — | — | — | pending |
| `special_xlogy_` | 1.004× | — | — | — | — | pending |
| `hardswish` | 0.882× | — | — | — | — | pending |
| `fmod_tensor_` | 0.818× | — | — | — | — | pending |
| `batch_norm_with_update_functional` | 1.748× | — | — | — | — | pending |
