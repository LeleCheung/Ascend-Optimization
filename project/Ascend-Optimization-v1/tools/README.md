# 公共工具

本目录保存多算子复用的环境、评测和分析工具。只服务 narrow_copy 的脚本已集中到 [该算子目录](../operators/narrow_copy/README.md)。

| 目录 | 用途 | 主要入口 |
|---|---|---|
| `environment/` | Claude 包装、配置、Python 探针、KGS 启动 | `start-kgs-19654.sh`、`bin-claude` |
| `evaluation/` | KG 运行、FlagGems 对照、历史汇总 | `run-kg-operator-910b.sh`、`compare-flaggems-910b.py`、`summarize-flaggems-comparison.py` |
| `profiling/` | 独立证据分析与体检报告 | `ascend_profiling_workflow.py`、`ascend_profiling_agent.py` |
| `tests/` | 证据绑定和拒绝条件验证 | `test_ascend_profiling_workflow.py`、`test_ascend_profiling_agent.py` |

设备实验在 910B 执行，纯 JSON 汇总与证据验证可在 Windows 执行。各运行工具仍包含服务器实际环境路径，运行前检查 KGS、模型、Catalog、设备和工作区。`run-kg-operator-910b.sh` 默认两轮并带 `--skip-review`，正式实验应按项目计划明确预算与审核策略。

`compare-flaggems-910b.py` 当前内置五个历史算子与对应服务器运行目录，是联合复测工具，不会自动完成新目标的 master 基线或三版本闭环。`tests/verify-phase1-case-api-910b.sh` 也只用于旧阶段验证。

三项历史基线的最小修复由 `evaluation/repair-flaggems-baselines.py` 生成；`prepare-repaired-flaggems-inputs.py` 准备已归档合同和候选；`compare-repaired-flaggems-910b.py` 在专用单卡 KGS 上复测；`assemble-repaired-flaggems.py` 保留首批失败证据并汇入完成的复测批次；`summarize-repaired-flaggems.py` 校验完整正确性后计算直接性能比，记录见 [修复复测](../archive/comparisons/flaggems-repaired-comparison-20261009/README.md)。PReLU 精度诊断工具为 `diagnose-prelu-reduction-910b.py`，输入源码和诊断结果随该报告归档。

PReLU 修复生成器默认依赖同目录的 `prelu-compensated-reduction.py`，补齐 FP32 误差补偿归约；`--prelu-reduction staged` 可复现轻量分阶段归约试验。旧候选偶发 FP32 精度失败，新候选由 `repair-prelu-candidate.py` 结合 `prelu-accurate-fused.py` 生成，融合输入梯度与稳定部分归约。原候选及各次复测均保留。

`evaluation/audit-repaired-flaggems.py` 从原始请求重算三轮结果、逐项核验 108 行 CSV，并重建源码、检查环境快照与文档链接。生成源码与早期测量文件的比较仅归一化换行；测量源码 SHA 仍按原字节严格验证。

在仓库根目录验证已有分析器：

```bash
python -B -m unittest discover -s project/Ascend-Optimization-v1/tools/tests -p 'test_ascend_profiling_*.py'
```

汇总历史联合对照：

```bash
python -B project/Ascend-Optimization-v1/tools/evaluation/summarize-flaggems-comparison.py project/Ascend-Optimization-v1/archive/comparisons/flaggems-comparison-20261009
```

汇总脚本会重写该批次的汇总文件，验证时可先复制至临时目录。原始数据归档保持原样。

`--no-profile` 关闭优化反馈，设备计时仍可能使用 NPU profiler。当前独立分析器已用历史证据测试；Roofline 尚未获得可用结果。三项修复报告、归档索引与所需工具在本次远端实验目录的 delivery/ 下提供独立快照；服务器 Git checkout 的全量目录迁移另行处理。
