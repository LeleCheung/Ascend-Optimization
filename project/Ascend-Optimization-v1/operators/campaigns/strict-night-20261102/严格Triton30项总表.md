# 严格Triton夜间优化30项总表

冻结起点16/30；本轮已核验25/30。队列：running。
加速比为KGS完整调用；CANN参考类型沿用冻结审计，PyTorch另列。待核验候选不计数。

## 1015（7/10）

|算子|起点相对CANN|HG-KDA相对CANN|相对PyTorch|原合同|状态|
|---|---:|---:|---:|---|---|
|[selu_](../strict-triton-20261102/candidates/selu_/restored-v1/main.py)|0.982×|0.982×|0.922×|36/36|严格Triton达标|
|[rad2deg_](proposals/rad2deg_/codex-launch-once-v1/main.py)|0.767×|0.922×|1.119×|36/36|严格Triton达标|
|[log10_](../strict-triton-20261102/candidates/log10_/restored-v1/main.py)|0.843×|0.843×|0.806×|36/36|严格Triton达标|
|[cumulative_trapezoid](../strict-triton-20261102/candidates/cumulative_trapezoid/restored-v1/main.py)|—|—|0.845×|33/33|原语义或补测未通过|
|[uniform](../strict-triton-20261102/candidates/uniform/restored-v1/main.py)|—|—|1.387×|18/18|原语义通过，CANN参考不合格或缺失，单列|
|[value_selecting_reduction_backward](../strict-triton-20261102/candidates/value_selecting_reduction_backward/restored-v1/main.py)|1.470×|1.470×|1.676×|96/96|严格Triton达标|
|[scatter_add](../strict-triton-20261102/candidates/scatter_add/restored-v1/main.py)|1.886×|1.886×|1.880×|18/18|严格Triton达标|
|[upsample_bilinear2d](../strict-triton-20261102/candidates/upsample_bilinear2d/restored-v1/main.py)|1.147×|1.147×|0.945×|24/24|严格Triton达标|
|[fractional_max_pool3d](../strict-triton-20261102/candidates/fractional_max_pool3d/restored-v1/main.py)|—|—|7.404×|9/9|原语义通过，CANN参考不合格或缺失，单列|
|[_conj_physical](../strict-triton-20261102/candidates/_conj_physical/restored-v1/main.py)|1.153×|1.153×|45.279×|14/14|严格Triton达标|

## 1029（18/20）

|算子|起点相对CANN|HG-KDA相对CANN|相对PyTorch|原合同|状态|
|---|---:|---:|---:|---|---|
|[negative_](../strict-triton-20261102/candidates/negative_/restored-v1/main.py)|0.814×|0.814×|0.560×|30/30|严格Triton达标|
|[xlogy_scalar_tensor](../strict-triton-20261102/candidates/xlogy_scalar_tensor/restored-v1/main.py)|1.579×|1.579×|1.234×|30/30|严格Triton达标|
|[nan_to_num_](../strict-triton-20261102/candidates/nan_to_num_/restored-v1/main.py)|1.100×|1.100×|0.876×|30/30|严格Triton达标|
|[cross_out](../strict-triton-20261102/candidates/cross_out/restored-v1/main.py)|4.090×|4.090×|3.757×|21/21|严格Triton达标|
|[expm1_](../strict-triton-20261102/candidates/expm1_/restored-v1/main.py)|—|—|0.551×|30/30|原语义通过，CANN参考不合格或缺失，单列|
|[hardshrink_backward](proposals/hardshrink_backward/codex-npu-masked-gradient-v1/main.py)|0.787×|1.042×|1.037×|30/30|严格Triton达标|
|[subtract_](../strict-triton-20261102/candidates/subtract_/restored-v1/main.py)|1.556×|1.556×|1.131×|30/30|严格Triton达标|
|[native_group_norm](../strict-triton-20261102/candidates/native_group_norm/restored-v1/main.py)|0.987×|0.987×|0.783×|27/27|严格Triton达标|
|[log2_](proposals/log2_/codex-transfer-log10-v2/main.py)|0.633×|0.846×|0.634×|30/30|严格Triton达标|
|[transform_bias_rescale_qkv](../strict-triton-20261102/candidates/transform_bias_rescale_qkv/reviewed-cpp-host-v2/main.py)|0.931×|0.931×|1.043×|21/21|严格Triton达标|
|[fix](candidates/fix/b8fe66ad0725e7e8/main.py)|0.789×|1.014×|0.735×|30/30|严格Triton达标|
|[round_out](proposals/round_out/codex-optional-out-v3/main.py)|0.585×|0.829×|0.942×|30/30|严格Triton达标|
|[hardswish_out](proposals/hardswish_out/codex-vectorcall-out-v4/main.py)|0.637×|0.862×|1.050×|30/30|严格Triton达标|
|[fake_quantize_per_tensor_affine_cachemask_out](../strict-triton-20261102/candidates/fake_quantize_per_tensor_affine_cachemask_out/restored-v1/main.py)|7.060×|7.060×|9.623×|30/30|严格Triton达标|
|[special_round_out](proposals/special_round_out/codex-required-out-v4/main.py)|0.613×|0.946×|0.967×|30/30|严格Triton达标|
|[special_xlogy_](../strict-triton-20261102/candidates/special_xlogy_/restored-v1/main.py)|1.004×|1.004×|0.846×|30/30|严格Triton达标|
|[hardswish](candidates/hardswish/581ad44a1694f431/main.py)|0.799×|0.882×|0.731×|30/30|严格Triton达标|
|[fmod_tensor_](proposals/fmod_tensor_/codex-streaming-fmod-v1/main.py)|0.727×|0.818×|0.729×|30/30|严格Triton达标|
|[linalg_inv_ex](../strict-triton-20261102/candidates/linalg_inv_ex/restored-v1/main.py)|—|—|—|5/7|原语义或补测未通过|
|[batch_norm_with_update_functional](../strict-triton-20261102/candidates/batch_norm_with_update_functional/restored-v1/main.py)|1.748×|1.748×|23.563×|27/27|严格Triton达标|

参考不合格或缺失的算子单列；`fractional_max_pool3d`和`_conj_physical`的PyTorch参考包含CPU回退。
完整源码身份、原始测量、补测、归属、参考类型和后端证明见`delivery.json`；本脚本不触发任何设备评测。
