# CANN优先30项对照

固定FlagGems d6a8eec、FlagTree 0.6.0+ascend.gitc286cba6、CANN 9.0.0。
原PyTorch参考常规达标24/30；本批已核验CANN参考7项、相对CANN达标7项。
本批使用KGS完整调用计时；CANN兼容组合包含转换与状态适配成本。历史列仅保留前批相同源码的结果。

|算子|CANN参考|本批相对CANN|本批相对PyTorch|历史相对PyTorch|状态|
|---|---|---|---|---|---|
|selu_|原生CANN|0.982×|—|0.904×|既有源码CANN对照达标|
|rad2deg_|原生CANN|—|—|1.039×|CANN参考待完成或未通过资格|
|log10_|原生CANN|—|—|0.938×|CANN参考待完成或未通过资格|
|cumulative_trapezoid|未核验|—|—|—|尚无合格同语义CANN参考|
|uniform|未核验|—|—|1.310×|尚无合格同语义CANN参考|
|value_selecting_reduction_backward|未核验|—|—|1.479×|尚无合格同语义CANN参考|
|scatter_add|原生CANN|—|—|1.853×|CANN参考待完成或未通过资格|
|upsample_bilinear2d|原生CANN|—|—|0.954×|CANN参考待完成或未通过资格|
|fractional_max_pool3d|未核验|—|—|—|尚无合格同语义CANN参考|
|_conj_physical|未核验|—|—|—|尚无合格同语义CANN参考|
|negative_|原生CANN|1.251×|—|0.937×|既有源码CANN对照达标|
|xlogy_scalar_tensor|未核验|—|—|1.227×|尚无合格同语义CANN参考|
|nan_to_num_|原生CANN|1.100×|—|0.866×|既有源码CANN对照达标|
|cross_out|原生CANN|—|—|3.891×|CANN参考待完成或未通过资格|
|expm1_|CANN＋dtype兼容|—|—|0.879×|CANN参考待完成或未通过资格|
|hardshrink_backward|原生CANN|—|—|0.901×|CANN参考待完成或未通过资格|
|subtract_|原生CANN|—|—|1.145×|CANN参考待完成或未通过资格|
|native_group_norm|原生CANN|1.024×|—|0.814×|既有源码CANN对照达标|
|log2_|原生CANN|1.055×|—|0.915×|既有源码CANN对照达标|
|transform_bias_rescale_qkv|原生CANN|—|—|0.916×|CANN参考待完成或未通过资格|
|fix|原生CANN|—|—|0.862×|CANN参考待完成或未通过资格|
|round_out|原生CANN|—|—|1.412×|CANN参考待完成或未通过资格|
|hardswish_out|原生CANN|1.362×|1.463×|0.871×|新候选CANN对照达标|
|fake_quantize_per_tensor_affine_cachemask_out|未核验|—|—|—|尚无合格同语义CANN参考|
|special_round_out|原生CANN|—|—|1.247×|CANN参考待完成或未通过资格|
|special_xlogy_|未核验|—|—|0.842×|尚无合格同语义CANN参考|
|hardswish|原生CANN|1.061×|—|0.905×|既有源码CANN对照达标|
|fmod_tensor_|原生CANN|—|—|1.242×|CANN参考待完成或未通过资格|
|linalg_inv_ex|未核验|—|—|—|尚无合格同语义CANN参考|
|batch_norm_with_update_functional|未核验|—|—|—|尚无合格同语义CANN参考|

本批新跨过CANN验收线：hardswish_out。
待返回正式请求1项；证据核验148项。
原参考含CPU回退、确定性条件及求逆奇异语义继续单列；不将30项写成默认全部达标。
