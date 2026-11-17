# 轨迹接续与性能恢复清单

下降仅表示固定参考下的历史耗时差异，尚不等于同环境的代码负优化。新实验与历史记录分别保存。

| 算子 | 待补连接 | 下降段 |
| --- | ---: | --- |
| selu_ | 3 | 无 |
| rad2deg_ | 3 | 0.9558 → 0.8344× |
| log10_ | 4 | 1.1000 → 0.9409× |
| uniform | 1 | 无 |
| value_selecting_reduction_backward | 2 | 无 |
| scatter_add | 2 | 无 |
| upsample_bilinear2d | 2 | 1.1754 → 1.1423× |
| _conj_physical | 1 | 1.5850 → 1.2344× |
| negative_ | 2 | 无 |
| xlogy_scalar_tensor | 0 | 无 |
| nan_to_num_ | 1 | 无 |
| cross_out | 0 | 无 |
| hardshrink_backward | 4 | 1.0533 → 0.9986× |
| subtract_ | 0 | 无 |
| native_group_norm | 1 | 无 |
| log2_ | 1 | 无 |
| transform_bias_rescale_qkv | 1 | 无 |
| fix | 0 | 无 |
| round_out | 3 | 1.0087 → 0.9165× |
| hardswish_out | 2 | 无 |
| fake_quantize_per_tensor_affine_cachemask_out | 0 | 无 |
| special_round_out | 3 | 无 |
| special_xlogy_ | 1 | 无 |
| hardswish | 3 | 0.8548 → 0.8492× |
| fmod_tensor_ | 2 | 无 |
| batch_norm_with_update_functional | 0 | 1.8126 → 1.7739× |
