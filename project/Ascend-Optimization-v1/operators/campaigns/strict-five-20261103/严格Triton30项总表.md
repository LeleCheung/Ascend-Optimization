# 严格Triton30项总表

当前25/30：1015为7/10，1029为18/20。验收要求为相对合格CANN／Ascend C参考超过0.8×，并通过原正确性、补测、人工审查及实际Triton后端核验。

所有成绩均来自KGS完整调用walltime。已达标25项沿用冻结源码和原测量；本轮仅更新expm1_的合格候选，未新增CANN达标项。

## 1015

|算子|相对CANN／Ascend C|相对PyTorch|完整评测|状态|
|---|---:|---:|---|---|
|`selu_`|0.982×|0.922×|36/36|严格Triton达标|
|`rad2deg_`|0.922×|1.119×|36/36|严格Triton达标|
|`log10_`|0.843×|0.806×|36/36|严格Triton达标|
|`cumulative_trapezoid`|—|0.845×|33/33|原语义或补测未通过|
|`uniform`|—|1.628×|18/18|原语义通过，CANN参考不合格或缺失，单列|
|`value_selecting_reduction_backward`|1.470×|1.676×|96/96|严格Triton达标|
|`scatter_add`|1.886×|1.880×|18/18|严格Triton达标|
|`upsample_bilinear2d`|1.147×|0.945×|24/24|严格Triton达标|
|`fractional_max_pool3d`|—|33.002×|9/9|原语义通过，CANN参考不合格或缺失，单列|
|`_conj_physical`|1.153×|45.279×|14/14|严格Triton达标|

## 1029

|算子|相对CANN／Ascend C|相对PyTorch|完整评测|状态|
|---|---:|---:|---|---|
|`negative_`|0.814×|0.560×|30/30|严格Triton达标|
|`xlogy_scalar_tensor`|1.579×|1.234×|30/30|严格Triton达标|
|`nan_to_num_`|1.100×|0.876×|30/30|严格Triton达标|
|`cross_out`|4.090×|3.757×|21/21|严格Triton达标|
|`expm1_`|—|0.647×|30/30|原PyTorch正确；性能不足且CANN参考未合格|
|`hardshrink_backward`|1.042×|1.037×|30/30|严格Triton达标|
|`subtract_`|1.556×|1.131×|30/30|严格Triton达标|
|`native_group_norm`|0.987×|0.783×|27/27|严格Triton达标|
|`log2_`|0.846×|0.634×|30/30|严格Triton达标|
|`transform_bias_rescale_qkv`|0.931×|1.043×|21/21|严格Triton达标|
|`fix`|1.014×|0.735×|30/30|严格Triton达标|
|`round_out`|0.829×|0.942×|30/30|严格Triton达标|
|`hardswish_out`|0.862×|1.050×|30/30|严格Triton达标|
|`fake_quantize_per_tensor_affine_cachemask_out`|7.060×|9.623×|30/30|严格Triton达标|
|`special_round_out`|0.946×|0.967×|30/30|严格Triton达标|
|`special_xlogy_`|1.004×|0.846×|30/30|严格Triton达标|
|`hardswish`|0.882×|0.731×|30/30|严格Triton达标|
|`fmod_tensor_`|0.818×|0.729×|30/30|严格Triton达标|
|`linalg_inv_ex`|—|77.743×|7/7|原语义通过，CANN参考不合格或缺失，单列|
|`batch_norm_with_update_functional`|1.748×|23.563×|27/27|严格Triton达标|

`fractional_max_pool3d`和`linalg_inv_ex`的PyTorch参考包含CPU回退，数值不能作为相对原生CANN的性能结论。累计积分仍有FP16补测失败。源码SHA、参考类型及原始证据路径见delivery.json。
