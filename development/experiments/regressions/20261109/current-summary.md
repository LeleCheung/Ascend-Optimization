# 当前算子验收状态

原合同及语义口径达标 **26/30**；累计积分独立数学参考合同另列 1 项。
原 25 项固定候选回归为 23/25；随后新候选和新参考分别记录。全部正式性能采用 KGS 完整调用 walltime。

## 1015

| 算子 | 相对 CANN | 完整评测 | 状态 |
| --- | ---: | --- | --- |
| `selu_` | 1.039× | 36/36 | 达标 |
| `rad2deg_` | 0.834× | 36/36 | 达标 |
| `log10_` | 0.941× | 36/36 | 达标 |
| `cumulative_trapezoid` | 1.012× | 33/33 | 独立数学合同通过；原默认合同未闭环 |
| `uniform` | 2.399× | 18/18 | 达标 |
| `value_selecting_reduction_backward` | 1.349× | 96/96 | 达标 |
| `scatter_add` | 1.892× | 18/18 | 达标 |
| `upsample_bilinear2d` | 1.142× | 24/24 | 达标 |
| `fractional_max_pool3d` | — | — | CANN参考未合格 |
| `_conj_physical` | 1.234× | 14/14 | 达标 |

## 1029

| 算子 | 相对 CANN | 完整评测 | 状态 |
| --- | ---: | --- | --- |
| `negative_` | 0.806× | 30/30 | 达标 |
| `xlogy_scalar_tensor` | 1.536× | 30/30 | 达标 |
| `nan_to_num_` | 0.863× | 30/30 | 达标 |
| `cross_out` | 3.868× | 21/21 | 达标 |
| `expm1_` | — | — | CANN参考未合格 |
| `hardshrink_backward` | 0.999× | 30/30 | 达标 |
| `subtract_` | 1.659× | 30/30 | 达标 |
| `native_group_norm` | 1.004× | 27/27 | 达标 |
| `log2_` | 0.880× | 30/30 | 达标 |
| `transform_bias_rescale_qkv` | 0.941× | 21/21 | 达标 |
| `fix` | 0.969× | 30/30 | 达标 |
| `round_out` | 0.916× | 30/30 | 达标 |
| `hardswish_out` | 0.867× | 30/30 | 达标 |
| `fake_quantize_per_tensor_affine_cachemask_out` | 8.155× | 30/30 | 达标 |
| `special_round_out` | 1.000× | 30/30 | 达标 |
| `special_xlogy_` | 1.071× | 30/30 | 达标 |
| `hardswish` | 0.849× | 30/30 | 达标 |
| `fmod_tensor_` | 0.841× | 30/30 | 达标 |
| `linalg_inv_ex` | — | — | CANN参考未合格 |
| `batch_norm_with_update_functional` | 1.774× | 27/27 | 达标 |
