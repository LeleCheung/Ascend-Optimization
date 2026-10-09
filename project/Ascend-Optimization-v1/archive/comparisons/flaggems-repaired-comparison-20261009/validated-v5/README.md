# 三项修复基线与候选：最终复测

三项双方均完整通过三轮 KGS 评测。

| 算子 | 修复基线 / PyTorch | 候选 / PyTorch | 候选 / 修复版 FlagGems | 正确性 / 计时（每轮双方） |
|---|---:|---:|---:|---|
| PReLU backward | 0.49× | 0.95× | **1.97×** | 21/21；9/9 |
| BatchNorm backward | 2.32× | 5.01× | **2.19×** | 30/30；15/15 |
| SmoothL1 backward | 2.17× | 3.48× | **1.61×** | 291/291；12/12 |

## 三轮稳定性

| 算子 | 第 1 轮 | 第 2 轮 | 第 3 轮 | 更快的计时用例数（每轮） |
|---|---:|---:|---:|---|
| PReLU backward | 1.9742× | 1.9955× | 1.9700× | 7/9、7/9、7/9 |
| BatchNorm backward | 2.1896× | 2.1961× | 2.1694× | 13/15、13/15、13/15 |
| SmoothL1 backward | 1.5994× | 1.6167× | 1.6089× | 12/12、12/12、12/12 |

## 计时、版本和证据

每轮交替基线/候选顺序，专用 KGS 端口 19656，仅暴露物理 NPU 7（API 中为 npu:0），串行执行。FlagGems adapter 使用设备侧 kernel 计时；本报告不测完整 Python 调用耗时。直接加速比由相同 workload 的基线延迟除以候选延迟计算，完整数据见 [comparison.json](comparison.json)、[per-case.csv](per-case.csv)。

历史原文件锁定提交 `4772d816bc5d52d37c4718f36adf377d81261f83`；执行测试的 checkout 为 `349011a9cd4b4180ad170515918dff7ff338e346`。当前报告基线是历史源码的本机修复版，修复源码以 KGS override 注入；共享 checkout 未修改。源码与补丁见 [repairs/](repairs/)，逐请求、源码摘要、合同和完整结果见 [measurements/](measurements/)，完整 workload 列表见 [protocol.json](protocol.json)。

PReLU 基线保留原 1024 分块逐元素内核及临时缓冲，补齐通道索引和权重梯度归约，FP32 使用 TwoSum 误差补偿以通过原容差。新候选将误差补偿的部分归约融合到输入梯度计算，低精度路径保持旧候选。基线补入稳定归约的额外开销全部计入，直接倍数对应此修复方案；表格同时列出双方相对 PyTorch 的速度。BatchNorm 在原内核按保存方差计算 rsqrt(var+eps)。SmoothL1 仅修正 beta=0、相等输入时梯度为 0。BatchNorm/SmoothL1 候选源码未改，PReLU 候选本轮增加了 FP32 精度修复，关键变化见 [周报](../周报精简版.md)。

PReLU 首批 BF16 失败保留在 [首批测量](../measurements/_prelu_kernel_backward/)，第二版超大 FP32 失败保留在 [FP32 归约批次](../followup-prelu-fp32sum/)。[补偿基线配旧候选批次](../followup-prelu-compensated/)三轮通过；[轻量归约配旧候选批次](../followup-prelu-staged/)第二轮发现旧候选 FP32 精度失败，首轮 2.11× 未采用。精度诊断见 [诊断报告](../diagnostics/README.md)。PReLU 最终数据来自 [新候选批次](../followup-prelu-accurate/)，BatchNorm/SmoothL1 来自首批；汇入关系、设备和批次记录见 [experiment.json](measurements/experiment.json)。

## 从现有证据复核

在仓库根目录执行：

```bash
python project/Ascend-Optimization-v1/tools/evaluation/summarize-repaired-flaggems.py project/Ascend-Optimization-v1/archive/comparisons/flaggems-repaired-comparison-20261009/validated-v5
```

重新运行设备实验见 [总入口](../README.md)；每次使用新的输出目录，保持相同测试源码和 fingerprint。

交付核验见 [delivery-audit.json](delivery-audit.json)。在仓库根运行以下命令，可重新核验完整三轮、108 行延迟、源码生成步骤和文档链接；服务退出与共享源码不变依据归档的 environment-current.json，命令本身不连接服务器。

```bash
python project/Ascend-Optimization-v1/tools/evaluation/audit-repaired-flaggems.py project/Ascend-Optimization-v1/archive/comparisons/flaggems-repaired-comparison-20261009
```
