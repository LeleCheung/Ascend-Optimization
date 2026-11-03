# HG-KDA / Ascend-Optimization

HG-KDA 面向昇腾算子的性能诊断与严格 Triton 优化，将成对 profiling、源码与实际 IR、数值参考、指令热点及分类知识组织为可追溯的模型上下文。

实现由 **skill、独立 Python 核心与可选后端适配器** 组成。模型由调用宿主提供，KG 可作为优化执行器；KGS 负责完整评测，msprof 负责采集。离线核心不依赖 KG/KGS 或 NPU 环境。

## 安装与快速验证

需要 Python 3.10 或更高版本。在仓库根目录执行：

    git lfs pull
    python -m pip install .
    python -m hgkda doctor
    python skills/hgkda/scripts/run_hgkda.py doctor
    python -m unittest discover -s tests -v

分析随仓库提供的真实历史证据，无需连接设备：

    hgkda compare --directory examples/paired-expm1/paired --output ../hgkda-output/compare.json
    hgkda context --operator expm1_ --source examples/paired-expm1/candidate.py --comparison ../hgkda-output/compare.json --output ../hgkda-output/context.json
    hgkda hotspots --listing examples/matmul-instructions/listing.json --source examples/matmul-instructions/candidate.py --output ../hgkda-output/hotspots.json
    hgkda regions --listing examples/matmul-instructions/listing.json --source-map examples/matmul-instructions/source-map.json --output ../hgkda-output/regions.json

输出文件不能覆盖；再次运行时使用新目录。该示例复算已有采集记录，不代表重新运行设备实验或新增性能成果。

## 工作流

固定源码与合同 → 成对采集与对照 → 按需检查源码/IR、数值参考和热点区域 → 模型提出可验证修改 → 严格 Triton 审查 → KGS 完整评测 → 记录结果与更新知识。

事实、规则假设和验证结论分别保存。活跃阈值仅提供排查优先级；经验吞吐包络不等于物理峰值；指令时间和不同流水时间不能累加为完整调用耗时。

## 目录

| 目录 | 内容 |
| --- | --- |
| skills/hgkda/ | 可调用 skill、执行入口及接口说明 |
| src/hgkda/ | 标准库分析核心、知识和可选适配器 |
| examples/ | 成对指标、指令区域及 KGS 评测示例 |
| benchmarks/ | 暂停时三十项算子的源码与结果索引 |
| tests/ | 身份、统计、时间轴与适配边界验证 |
| docs/、manifests/ | 方法、配置、来源及恢复记录 |
| sources/、third_party/ | 固定的可选开发依赖；不参与核心包安装 |
| tools/ | 维护工具及参考后端构建工具 |

## 实验边界

候选全部设备计算和搬运使用 Triton；CANN 原生库或来源明确的 Ascend C 实现作为参考。PyTorch 与 CANN 参考结果分别记录。源码审查、正确性、性能和附加语义检查不能互相替代。

暂停时严格验收为 **25/30**，详见 benchmarks/index.json。本次工程整理不恢复旧优化，也不把包安装或接口验证计为新增达标。原始实验保留在 Git 检查点 f63e6c30a0d8583ba092844e57ab90c0f6d0349f，精选文件保留来源与 SHA。

## 文档

- [skill 接口](skills/hgkda/references/interface.md)
- [可选后端](docs/backends.md)
- [目录与调用](docs/directory-layout.md)
- [工程整理计划](docs/release-plan.md)
- [验证结果](docs/validation.md)
- [历史恢复](docs/recovery.md)
- [维护约定](AGENTS.md)

本仓库当前仅同步既有私有远端，不执行公开发布或 Release。第三方版权和许可证按来源保留；公开发布前应独立核对可分发范围。
