# 三项历史 FlagGems 修复与性能复测

PReLU backward、BatchNorm backward、SmoothL1 backward 已完成本机正确性修复和三轮对照。双方每轮均通过完整 KGS workload，在同一张 910B 物理 NPU 7 上串行运行。

| 算子 | 修复基线 / PyTorch | 候选 / PyTorch | 候选 / 修复版 FlagGems | 正确性 / 计时（每轮双方） |
|---|---:|---:|---:|---|
| PReLU backward | 0.49× | 0.95× | **1.97×** | 21/21；9/9 |
| BatchNorm backward | 2.32× | 5.01× | **2.19×** | 30/30；15/15 |
| SmoothL1 backward | 2.17× | 3.48× | **1.61×** | 291/291；12/12 |

PReLU 旧候选约 2.8×，重复测试出现 FP32 精度失败；新候选修复归约精度后性能下降，当前略慢于 PyTorch。BatchNorm 和 SmoothL1 保持完整正确性并超过 PyTorch。三项属于历史流程与方法验证；新主线仍按 Excel 华为列低于 0.8×、正确性通过筛选。

可直接汇报的文字见 [周报精简版](周报精简版.md)。三轮明细见 [最终报告](validated-v5/README.md)、[汇总 JSON](validated-v5/comparison.json)、[108 行逐 case 延迟](validated-v5/per-case.csv)。

按目标逐项核验的证据见 [交付核验](交付核验.md)，910B 独立快照位置与同步状态见 [delivery-sync.json](delivery-sync.json)。

## 修复与接口适配

| 对象 | 历史问题 | 最终处理 |
|---|---|---|
| PReLU backward | 按最后一维索引，返回与输入同形状的权重梯度；本机 NPU 接口需要通道维索引和归约后的权重梯度 | 保留原 1024 分块逐元素内核，修复 dim=1 通道索引，补齐权重梯度归约；低精度乘积保持原 dtype、FP32 累加；FP32 标量归约加入 TwoSum 误差补偿 |
| BatchNorm backward | 原实现将保存统计量当作标准差倒数，本机 torch_npu 接口保存的是方差 | 原内核改为 rsqrt(var+eps)，保留 autotune；明确训练/推理统计量与输出掩码 |
| SmoothL1 backward | beta=0 且输入等值时返回 NaN，本机参考为 0 | 等值梯度改为 0，保留原分块和其余计算 |

基线是**历史 FlagGems 的本机正确性修复版**。历史原文件锁定提交 `4772d816bc5d52d37c4718f36adf377d81261f83`，执行测试的 checkout 为 `349011a9cd4b4180ad170515918dff7ff338e346`。原版、最终修复源码、补丁和 SHA 在 [最终 repairs](validated-v5/repairs/)；本机共享 checkout 未修改，修复通过 KGS implementation override 注入。

PReLU 基线新增的稳定归约及临时缓冲开销全部计入计时，1.97× 对应该修复方案。新候选融合输入梯度与补偿部分归约，低精度路径沿用旧候选。BatchNorm/SmoothL1 候选本轮未改。

双方共同使用历史测试合同：PReLU 已按通道维适配，BatchNorm 的保存方差取正值，benchmark 使用 case API。合同源码与差异保留在 [历史 contracts](../flaggems-comparison-20261009/measurements/contracts/)，完整 workload 见 [最终 protocol](validated-v5/protocol.json)。

## 评测口径与原始证据

专用 KGS 端口为 19656，物理 NPU 7 在 API 中映射为 `npu:0`，max-workers=1；奇数轮先基线、偶数轮先候选。使用现有 FlagGems adapter 的设备侧 kernel 计时。

每轮按相同 case 的“修复基线延迟 / 候选延迟”取几何平均，最终报告三轮中位数与范围。完整正确性、UUID、phase、axes、binding/settings、benchmark fingerprint、源码 SHA 与设备通过核验后才汇总。双方相对 PyTorch 的几何均值不直接相除。

PReLU 最终数据来自 [followup-prelu-accurate](followup-prelu-accurate/)，其余两项来自 [首批 measurements](measurements/)；[汇入记录](validated-v5/measurements/experiment.json)记录批次与物理设备。所有本次专用服务已退出，共享源码与提交号核验未变，见 [环境核验](environment-current.json)。

## PReLU 历史尝试

失败请求、源码和原始结果均保留，测试容差未放宽。

| 阶段 | 结果 | 证据 |
|---|---|---|
| v1：原逐元素内核 + 默认 sum | 第二轮 BF16 权重梯度失败；首轮 2.21× 未采用 | [首批记录](measurements/_prelu_kernel_backward/) |
| v2：sum 指定 FP32 累加 | 低精度 seed 诊断通过；完整测试超大 FP32 用例失败 | [FP32 sum 批次](followup-prelu-fp32sum/) |
| v3：补偿基线 + 旧候选 | 三轮通过；基线归约开销大，5.79× 未作为最终结果 | [v3 汇总](validated-v3/README.md) |
| v4：轻量分阶段归约 + 旧候选 | 旧候选第二轮 FP32 失败；首轮 2.11× 未采用 | [staged 批次](followup-prelu-staged/) |
| v5：补偿基线 + 融合补偿新候选 | 双方三轮全部通过；最终直接 1.97×，候选相对 PyTorch 0.95× | [最终结果](validated-v5/README.md) |

原因与固定 seed 对照见 [精度诊断](diagnostics/README.md)。原版正确性失败证据继续保留在 [历史联合对照](../flaggems-comparison-20261009/README.md)。

## 复核已归档结果

在 Windows 或 910B 的仓库根目录执行，无需占用 NPU：

```bash
python project/Ascend-Optimization-v1/tools/evaluation/summarize-repaired-flaggems.py \
  project/Ascend-Optimization-v1/archive/comparisons/flaggems-repaired-comparison-20261009/validated-v5
```

该命令核验三轮完整结果与源码绑定，再重算 comparison.json 和 per-case.csv。

## 重新运行设备实验

需要已配置的 KGS 6.5.0、910B 容器 Python 3.11.15、torch_npu/Triton 和上述同一 FlagGems 测试 checkout。仅克隆源码不足以重建 CANN 和容器环境。先在仓库根生成输入，输出目录应使用新的实验路径：

```bash
python project/Ascend-Optimization-v1/tools/evaluation/repair-flaggems-baselines.py \
  --historical-report project/Ascend-Optimization-v1/archive/comparisons/flaggems-comparison-20261009 \
  --output /tmp/flaggems-repair-new/repairs --prelu-reduction compensated
python project/Ascend-Optimization-v1/tools/evaluation/prepare-repaired-flaggems-inputs.py \
  --historical-report project/Ascend-Optimization-v1/archive/comparisons/flaggems-comparison-20261009 \
  --repairs /tmp/flaggems-repair-new/repairs \
  --output /tmp/flaggems-repair-new/inputs
python project/Ascend-Optimization-v1/tools/evaluation/repair-prelu-candidate.py \
  --input /tmp/flaggems-repair-new/inputs/_prelu_kernel_backward/candidate.py \
  --output /tmp/flaggems-repair-new/prelu-accurate.py
cp /tmp/flaggems-repair-new/prelu-accurate.py /tmp/flaggems-repair-new/inputs/_prelu_kernel_backward/candidate.py
cp /tmp/flaggems-repair-new/inputs/protocol.json /tmp/flaggems-repair-new/protocol.json
```

两个生成器依赖同目录的 prelu-compensated-reduction.py；新候选还依赖 prelu-accurate-fused.py。完整复制 tools/evaluation/ 即可。

早期 BatchNorm/SmoothL1 修复文件经过 Windows 换行转换，归档保留其实际提交给 KGS 的字节与 SHA。当前生成器写 LF，代码在换行归一化后完全一致；若需原 SHA，直接使用 validated-v5/inputs/ 中的测量源码。

在 `tle_yy` 已加载 CANN 的运行环境中启动专用 KGS，确认端口空闲，并保持原测试 checkout：

```bash
ASCEND_RT_VISIBLE_DEVICES=7 \
KGS_FLAGGEMS_ROOT=/data/hanle/ascend-optimization/FlagGems \
/usr/local/python3.11.15/bin/python3.11 -c 'from kernelgen_server.server import main; main()' \
  --host 127.0.0.1 --backend npu --timing walltime \
  --max-workers 1 --port 19656 \
  --profile-artifact-root /tmp/flaggems-repair-new/profiles
```

上述命令中的 `--timing walltime` 是本次服务启动参数；FlagGems adapter 仍使用设备侧 kernel 计时。另一终端在同一容器中运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=7 /usr/local/python3.11.15/bin/python3.11 \
  project/Ascend-Optimization-v1/tools/evaluation/compare-repaired-flaggems-910b.py \
  --inputs /tmp/flaggems-repair-new/inputs \
  --output /tmp/flaggems-repair-new/measurements \
  --server http://127.0.0.1:19656 --repeats 3
python project/Ascend-Optimization-v1/tools/evaluation/summarize-repaired-flaggems.py /tmp/flaggems-repair-new
```

脚本会核验历史 fingerprint，自动上传 bundle；任一轮未通过则该算子不进入性能汇总。实验结束后停止自己启动的专用服务。

实际执行脚本保留在 [首批 executed-scripts](executed-scripts/) 和 [最终 PReLU launcher](followup-prelu-accurate/launch.py)。launcher 当时从已有进程读取环境，并依赖前一批 PID，属于执行快照；新实验应使用上面的独立启动命令。原始远端运行目录为 `/data/hanle/ascend-optimization/runtime/kg-controller/runs/flaggems-repaired-comparison-20261009/`。
