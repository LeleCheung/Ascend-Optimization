# 剩余算子的 CANN 参考可用性

本轮保持 CANN 9.0.0、FlagTree 及当前回归环境不变，额外检查公开源码。源码清点不等于已在 910B 编译或执行。

## 新增检查范围

公开镜像 [ops-nn](https://github.com/Atlas-Ascend-Fork/ops-nn) 和 [ops-math](https://github.com/Atlas-Ascend-Fork/ops-math) 的 README 指向厂商上游 `gitcode.com/cann/ops-nn`、`gitcode.com/cann/ops-math`。本轮检查镜像的两个固定版本范围：

| 仓库 | 配套版本 v9.0.0 | master 固定提交 |
| --- | --- | --- |
| ops-nn | `0413e303e6cd1986b592c3e039f2579ac0f10b62` | `d1efef7c30ca124dc30787a7db3c4479c8ffcc05`（2026-11-07） |
| ops-math | `ce515930d875e6fb0187195e0b328ab2def4bea5` | `aae0dcf058854f45e035909c6091a64d2c8b3910`（2026-11-07） |

递归清单均未截断。v9.0.0 两仓分别含 19035、12931 项，master 分别含 43576、25600 项；这些数量包含目录。名称检索覆盖 `fractional`、`inv_ex`、`lu_factor` 和 `linalg_lu`，没有找到对应路径。该结论限定于已检查的公开镜像与版本。

## 两个缺失的完整接口

**linalg_inv_ex：存在 Inverse，仍缺完整 info 语义。** 当前 master 的 [aclnn_inverse.h](https://github.com/Atlas-Ascend-Fork/ops-nn/blob/d1efef7c30ca124dc30787a7db3c4479c8ffcc05/index/matrix_inverse/op_host/op_api/aclnn_inverse.h) 只接收 `self`、`out` 和工作区／执行器，未提供 `info` 输出。[matrix_inverse.cpp](https://github.com/Atlas-Ascend-Fork/ops-nn/blob/d1efef7c30ca124dc30787a7db3c4479c8ffcc05/index/matrix_inverse/op_host/op_api/matrix_inverse.cpp) 通过 `ADD_TO_LAUNCHER_LIST_AICPU` 调度设备侧实现。设备上的 AiCPU 与 PyTorch 的宿主 CPU 回退是不同路径，但两者均不能补上该接口未返回的逐矩阵 LU 状态。API 调用返回码也不能替代 `inv_ex.info`。

**fractional_max_pool3d：尚未找到同语义公开入口。** 两个版本的目录未找到 fractional 对应实现。普通／自适应三维最大池化不能直接代替原合同中的随机窗口、固定核大小、`_random_samples` 和索引输出。已有纯 Triton 候选及其正确性证据继续保留，原 PyTorch 含 CPU 回退的速度单列。

## 其余三项的本轮结果

| 算子 | 原问题 | 本轮结果 |
| --- | --- | --- |
| `uniform` | 原生单接口低精度输出存在区间越界 | CANN Uniform＋Clamp 组合资格 31/31，正式 18/18，两侧补测各 31/31；原候选相对新参考 2.399× |
| `expm1_` | 原生参考的小值相对精度及零符号未合格 | 默认与官方 `high_precision` 模式均为 0/3 类型通过小值数学资格；不报告不合格参考的性能 |
| `cumulative_trapezoid` | 默认参考的 FP16 非末轴输出存在不稳定性 | 新数学参考合同下 33/33、两侧补测各 36/36，1.012×；原默认合同未被宣布通过 |

Expm1 探测先修复 GE/TBE 私有缓存目录未建立导致的初始化失败，再在全新目录执行两个模式。高精度选项和设备调用均返回成功，但 FP16 在约 `1e-6` 处相对误差约 5.88%，超过原 1.5% 限值；BF16／FP32 在 `1e-8` 等小值处仍可能返回零。环境错误与数值失败分别归档。已安装 SDK 含 `aclnnExpm1`／`aclnnInplaceExpm1`，问题是当前实测资格未通过。

所检查的最新公开 Expm1 源码含稳定多项式等实现，但相关目录和配置面向其他架构，尚未证明可用于 910B；不将源码存在解释为本机可用。源码版本、检索清单与 API 片段收录于[本批源码研究](../benchmarks/regressions/20261109/native-source-study.zip)。

原始源码、API 返回和递归清单保存在控制端本批研究记录中，包含 URL 与 SHA；不据此宣称全局不存在其他厂商实现。正式回归的状态与新参考验证结果见 [回归记录](regression-20261109.md)。
