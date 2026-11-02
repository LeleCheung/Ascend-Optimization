# CANN优先30项对照

固定FlagGems d6a8eec、FlagTree 0.6.0+ascend.gitc286cba6、CANN 9.0.0。
原PyTorch参考常规达标24/30；本批已核验CANN参考25项、相对CANN达标25项。
本批使用KGS完整调用计时；CANN兼容组合包含转换与状态适配成本。历史列仅保留前批相同源码的结果。

|算子|CANN参考|本批相对CANN|本批相对PyTorch|历史相对PyTorch|状态|
|---|---|---|---|---|---|
|selu_|原生CANN|0.982×|—|0.904×|既有源码CANN对照达标|
|rad2deg_|原生CANN|1.297×|1.506×|1.039×|新候选CANN对照达标|
|log10_|原生CANN|1.306×|—|0.938×|既有源码CANN对照达标|
|cumulative_trapezoid|CANN组合：加法＋缩放＋扫描|—|—|—|CANN参考补测未通过|
|uniform|原生CANN＋随机状态适配|—|—|1.310×|CANN参考补测未通过|
|value_selecting_reduction_backward|CANN组合：清零＋Scatter|1.470×|—|1.479×|既有源码CANN对照达标|
|scatter_add|原生CANN|1.886×|—|1.853×|既有源码CANN对照达标|
|upsample_bilinear2d|CANN＋布局兼容|1.147×|—|0.954×|既有源码CANN对照达标|
|fractional_max_pool3d|未核验|—|—|—|尚无合格同语义CANN参考|
|_conj_physical|CANN组合：复制＋虚部取负|1.343×|—|—|既有源码CANN对照达标|
|negative_|原生CANN|1.251×|—|0.937×|既有源码CANN对照达标|
|xlogy_scalar_tensor|原生CANN|1.579×|—|1.227×|既有源码CANN对照达标|
|nan_to_num_|原生CANN|1.100×|—|0.866×|既有源码CANN对照达标|
|cross_out|原生CANN|4.057×|—|3.891×|既有源码CANN对照达标|
|expm1_|CANN＋dtype兼容|—|—|0.879×|CANN参考补测未通过|
|hardshrink_backward|原生CANN|0.894×|—|0.901×|既有源码CANN对照达标|
|subtract_|原生CANN|1.556×|—|1.145×|既有源码CANN对照达标|
|native_group_norm|原生CANN|1.024×|—|0.814×|既有源码CANN对照达标|
|log2_|原生CANN|1.055×|—|0.915×|既有源码CANN对照达标|
|transform_bias_rescale_qkv|原生CANN|0.931×|1.043×|0.916×|新候选CANN对照达标|
|fix|原生CANN|1.197×|—|0.862×|既有源码CANN对照达标|
|round_out|原生CANN|1.430×|—|1.412×|既有源码CANN对照达标|
|hardswish_out|原生CANN|1.395×|1.708×|0.871×|新候选CANN对照达标|
|fake_quantize_per_tensor_affine_cachemask_out|CANN＋FP32转换|6.975×|—|—|既有源码CANN对照达标|
|special_round_out|原生CANN|1.261×|—|1.247×|既有源码CANN对照达标|
|special_xlogy_|原生CANN|1.004×|—|0.842×|既有源码CANN对照达标|
|hardswish|原生CANN|1.061×|—|0.905×|既有源码CANN对照达标|
|fmod_tensor_|原生CANN|1.422×|—|1.242×|既有源码CANN对照达标|
|linalg_inv_ex|未核验|—|—|—|尚无合格同语义CANN参考|
|batch_norm_with_update_functional|CANN＋FP32／状态适配|1.783×|—|—|既有源码CANN对照达标|

本批新跨过CANN验收线：rad2deg_、transform_bias_rescale_qkv、hardswish_out。
待返回正式请求0项；证据核验454项。
原参考含CPU回退、确定性条件及求逆奇异语义继续单列；不将30项写成默认全部达标。
