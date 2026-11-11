# 独立 HG-KDA 固定候选回归

状态：running；已处理 2/25；本轮达标 2 项。

候选源码、原输入、容差和计时参数固定；本轮每项一次完整 KGS 正式评测，原补测独立执行。

加速比为完整调用的几何平均；CANN 参考类型含原生单接口及已审核的同语义组合，逐项保存在 JSON 的 reference_kind 字段中。

| 算子 | 拆分前相对 CANN | 本轮相对 CANN | 完整评测 | PyTorch 语义补测 | CANN 语义补测 | 本轮状态 |
| --- | ---: | ---: | --- | --- | --- | --- |
| `selu_` | 0.982× | 1.039× | 36/36 | 42/42 passed | 42/42 passed | 达标 |
| `rad2deg_` | 0.922× | 0.834× | 36/36 | 42/42 passed | 42/42 passed | 达标 |
| `log10_` | 0.843× | — | — | — | — | running |
| `value_selecting_reduction_backward` | 1.470× | — | — | — | — | pending |
| `scatter_add` | 1.886× | — | — | — | — | pending |
| `upsample_bilinear2d` | 1.147× | — | — | — | — | pending |
| `_conj_physical` | 1.153× | — | — | — | — | pending |
| `negative_` | 0.814× | — | — | — | — | pending |
| `xlogy_scalar_tensor` | 1.579× | — | — | — | — | pending |
| `nan_to_num_` | 1.100× | — | — | — | — | pending |
| `cross_out` | 4.090× | — | — | — | — | pending |
| `hardshrink_backward` | 1.042× | — | — | — | — | pending |
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
