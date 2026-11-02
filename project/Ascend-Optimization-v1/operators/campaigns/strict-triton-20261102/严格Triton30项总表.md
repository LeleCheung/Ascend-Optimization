# 严格Triton 10＋20复核

当前严格Triton达标16/30；队列completed。
所有加速比为KGS完整调用；合格CANN单接口与兼容／组合分别标注。历史混合实现不计入。

## 1015（6/10）

|算子|候选后端|参考类型|相对CANN|相对PyTorch|原合同|补测|状态|
|---|---|---|---:|---:|---|---|---|
|[selu_](candidates/selu_/restored-v1/main.py)|Triton|CANN单接口|0.982×|0.922×|36/36|原：42/42；CANN：42/42|严格Triton达标|
|[rad2deg_](candidates/rad2deg_/restored-v1/main.py)|Triton|CANN单接口|0.767×|1.143×|36/36|原：42/42；CANN：42/42|严格Triton正确，性能未过0.8×|
|[log10_](candidates/log10_/restored-v1/main.py)|Triton|CANN单接口|0.843×|0.806×|36/36|原：66/66；CANN：66/66|严格Triton达标|
|[cumulative_trapezoid](candidates/cumulative_trapezoid/restored-v1/main.py)|Triton|CANN累计积分组合|—|0.845×|33/33|原：34/36；CANN：—|原语义或补测未通过|
|[uniform](candidates/uniform/restored-v1/main.py)|Triton|CANN随机状态适配|—|1.387×|18/18|原：31/31；CANN：—|原语义通过，CANN参考不合格或缺失，单列|
|[value_selecting_reduction_backward](candidates/value_selecting_reduction_backward/restored-v1/main.py)|Triton|CANN清零＋回填|1.470×|1.676×|96/96|原：96/96；CANN：96/96|严格Triton达标|
|[scatter_add](candidates/scatter_add/restored-v1/main.py)|Triton|CANN单接口|1.886×|1.880×|18/18|原：21/21；CANN：21/21|严格Triton达标|
|[upsample_bilinear2d](candidates/upsample_bilinear2d/restored-v1/main.py)|Triton|CANN布局适配|1.147×|0.945×|24/24|原：30/30；CANN：30/30|严格Triton达标|
|[fractional_max_pool3d](candidates/fractional_max_pool3d/restored-v1/main.py)|Triton|无同语义参考|—|7.404×|9/9|原：15/15；CANN：—|原语义通过，CANN参考不合格或缺失，单列|
|[_conj_physical](candidates/_conj_physical/restored-v1/main.py)|Triton|CANN复制＋取负|1.153×|45.279×|14/14|原：16/16；CANN：16/16|严格Triton达标|

## 1029（10/20）

|算子|候选后端|参考类型|相对CANN|相对PyTorch|原合同|补测|状态|
|---|---|---|---:|---:|---|---|---|
|[negative_](candidates/negative_/restored-v1/main.py)|Triton|CANN单接口|0.814×|0.560×|30/30|原：63/63；CANN：63/63|严格Triton达标|
|[xlogy_scalar_tensor](candidates/xlogy_scalar_tensor/restored-v1/main.py)|Triton|CANN单接口|1.579×|1.234×|30/30|原：72/72；CANN：72/72|严格Triton达标|
|[nan_to_num_](candidates/nan_to_num_/restored-v1/main.py)|Triton|CANN单接口|1.100×|0.876×|30/30|原：54/54；CANN：54/54|严格Triton达标|
|[cross_out](candidates/cross_out/restored-v1/main.py)|Triton|CANN单接口|4.090×|3.757×|21/21|原：45/45；CANN：45/45|严格Triton达标|
|[expm1_](candidates/expm1_/restored-v1/main.py)|Triton|CANN精度适配|—|0.551×|30/30|原：63/63；CANN：—|原语义通过，CANN参考不合格或缺失，单列|
|[hardshrink_backward](candidates/hardshrink_backward/restored-v1/main.py)|Triton|CANN单接口|0.787×|0.851×|30/30|原：45/45；CANN：45/45|严格Triton正确，性能未过0.8×|
|[subtract_](candidates/subtract_/restored-v1/main.py)|Triton|CANN单接口|1.556×|1.131×|30/30|原：72/72；CANN：72/72|严格Triton达标|
|[native_group_norm](candidates/native_group_norm/restored-v1/main.py)|Triton|CANN单接口|0.987×|0.783×|27/27|原：42/42；CANN：42/42|严格Triton达标|
|[log2_](candidates/log2_/restored-v1/main.py)|Triton|CANN单接口|0.633×|0.570×|30/30|原：36/36；CANN：36/36|严格Triton正确，性能未过0.8×|
|[transform_bias_rescale_qkv](candidates/transform_bias_rescale_qkv/reviewed-cpp-host-v2/main.py)|Triton＋C++宿主|CANN单接口|0.931×|1.043×|21/21|原：48/48；CANN：48/48|严格Triton达标|
|[fix](candidates/fix/restored-v1/main.py)|Triton|CANN单接口|0.789×|0.689×|30/30|原：39/39；CANN：39/39|严格Triton正确，性能未过0.8×|
|[round_out](candidates/round_out/restored-v1/main.py)|Triton|CANN单接口|0.585×|0.556×|30/30|原：48/48；CANN：48/48|严格Triton正确，性能未过0.8×|
|[hardswish_out](candidates/hardswish_out/restored-v1/main.py)|Triton|CANN单接口|0.637×|0.832×|30/30|原：42/42；CANN：42/42|严格Triton正确，性能未过0.8×|
|[fake_quantize_per_tensor_affine_cachemask_out](candidates/fake_quantize_per_tensor_affine_cachemask_out/restored-v1/main.py)|Triton|CANN FP32兼容组合|7.060×|9.623×|30/30|原：36/36；CANN：36/36|严格Triton达标|
|[special_round_out](candidates/special_round_out/restored-v1/main.py)|Triton|CANN单接口|0.613×|0.646×|30/30|原：48/48；CANN：48/48|严格Triton正确，性能未过0.8×|
|[special_xlogy_](candidates/special_xlogy_/restored-v1/main.py)|Triton|CANN单接口|1.004×|0.846×|30/30|原：45/45；CANN：45/45|严格Triton达标|
|[hardswish](candidates/hardswish/restored-v1/main.py)|Triton|CANN单接口|0.799×|0.629×|30/30|原：42/42；CANN：42/42|严格Triton正确，性能未过0.8×|
|[fmod_tensor_](candidates/fmod_tensor_/restored-v1/main.py)|Triton|CANN单接口|0.727×|0.742×|30/30|原：57/57；CANN：57/57|严格Triton正确，性能未过0.8×|
|[linalg_inv_ex](candidates/linalg_inv_ex/restored-v1/main.py)|Triton|无同语义参考|—|—|5/7|原：6/8；CANN：—|原语义或补测未通过|
|[batch_norm_with_update_functional](candidates/batch_norm_with_update_functional/restored-v1/main.py)|Triton|CANN FP32状态组合|1.748×|23.563×|27/27|原：36/36；CANN：36/36|严格Triton达标|

`fractional_max_pool3d`、`_conj_physical`的PyTorch参考含CPU回退；该列只保留原合同结果，不能解释为相对纯NPU实现的收益。
算子名链接本轮冻结源码；来源、审查、归属与SHA见`candidate-index.json`及每项`review.json`。完整原始成绩和复用标记见`delivery.json`。
所有门槛使用未舍入的原始加速比判断。参考资格、原正确性、补测、后端与编译路径任何一项未通过，均不计为本轮达标。

