# HG-KDA / Ascend-Optimization

HG-KDA 面向昇腾算子的性能诊断与严格 Triton 优化，将同输入候选与参考的 profiling、源码与实际 IR、Roofline 参考、指令热点及分类知识组织为可追溯的模型上下文，支持从性能差异到机制假设、代码修改和完整评测的优化流程。

实现由 **skill、独立 Python 核心与可选后端适配器** 组成。模型由调用宿主提供，KernelGen（KG）可作为优化执行器；KernelGen Server（KGS）负责完整评测，msprof 负责设备采集。离线分析只依赖 Python 标准库。

## 核心能力

| 能力 | 输出 |
| --- | --- |
| 成对 profiling 对比 | 两侧指标分布、计数总和、候选/参考比值及平均搬运粒度 |
| 分阶段诊断 | 基于算子特征和指标的机制假设、下一项取证或验证建议 |
| 指令与区域分析 | 候选 kernel 热点、源码区域，以及有时间轴时的流水活动与等待 |
| 数值参考 | 在工作量与采集口径匹配时计算 Roofline 参考，区分经验包络与物理峰值 |
| 上下文与知识管理 | 绑定源码身份的证据上下文、分类知识、经验去重与按条件检索 |
| 评测接入 | 严格 Triton 静态筛查、审查记录绑定与 KGS 完整评测 |
| 上游验收 | 对显式外部FlagGems工作区验证导入来源，运行原生正确性测试与core benchmark |
| 性能反馈 | 逐case几何平均损失、大小输入代表case，以及继续优化或达标决策 |

## 工作流

固定源码与测试合同 → 成对采集与对照 → 按需检查源码/IR、数值参考和热点区域 → 模型提出可验证修改 → 严格 Triton 审查 → KGS 完整评测 → 性能反馈 → 继续优化或保留结果、更新知识。

skill 指导模型选择和调用工具；Python 核心计算与组织证据；模型负责解释机制并生成候选。模型调用与多轮优化由宿主或 KG 编排。

## 安装

需要 Python 3.10 或更高版本。推荐在独立虚拟环境中安装；获取仓库后，在hgkda目录执行：

    git lfs pull
    python -m pip install .
    python -m hgkda doctor

Git LFS 用于获取示例合同包等原件。安装核心包不会安装 KG/KGS、FlagGems、模型客户端或 NPU 运行环境；设备采集与评测所需配置见 [可选后端](docs/backends.md)。

## 快速示例

以下命令分析仓库附带的真实历史采集，无需连接设备：

    hgkda compare --directory examples/paired-expm1/paired --output ../hgkda-output/compare.json
    hgkda context --operator expm1_ --source examples/paired-expm1/candidate.py --comparison ../hgkda-output/compare.json --output ../hgkda-output/context.json
    hgkda hotspots --listing examples/matmul-instructions/listing.json --source examples/matmul-instructions/candidate.py --output ../hgkda-output/hotspots.json
    hgkda regions --listing examples/matmul-instructions/listing.json --source-map examples/matmul-instructions/source-map.json --output ../hgkda-output/regions.json

compare.json 保存成对指标，context.json 将证据与分类知识组织为模型上下文；hotspots.json 和 regions.json 分别展示指令热点与源码区域。输出保留输入身份，供模型或其他应用读取，公共封装见 [JSON 合同](schemas/result.schema.json)。

已有输出文件不会覆盖，再次运行时使用新目录。更多示例与 KGS 评测方法见 [示例说明](examples/README.md)。

## 使用 skill

让支持 skill 的模型宿主加载 [HG-KDA skill](skills/hgkda/SKILL.md)，并使用安装了核心包的 Python 执行工具。统一脚本入口与 hgkda CLI 提供相同功能：

    python skills/hgkda/scripts/run_hgkda.py doctor

参数和输入格式见 [接口说明](skills/hgkda/references/interface.md)。工程结构与开发方法见 [公共运行接口](docs/runtime.md)。

## 适用范围与评测约定

- 当前设备侧实例与验证以 Ascend 910B 为主，msprof 报告格式和运行环境应与采集配置匹配。
- 优化候选的全部设备计算和搬运使用 Triton；CANN 原生库或来源明确的 Ascend C 实现用于参考，PyTorch 与 CANN 参考成绩分别记录。
- 正式性能采用 KGS 完整调用计时，并保留完整正确性和所需的附加语义检查；独立 profiling 与模拟器耗时单列。
- 指标阈值形成待验证假设。经验吞吐包络与物理峰值分别解释，不累加可重叠的流水或指令时间作为总耗时。

当前交付算子位于同级operators目录；以其index.json及各算子manifest.json为准。公开软件包不依赖内部开发档案。

## 设备与模型宿主

    python -m hgkda.runtime --help

公共runtime提供prepare、context、preflight、pair、simulate、generate、evaluate等显式动作。详细配置见[公共运行接口](docs/runtime.md)，外部服务见[后端说明](docs/backends.md)。宿主负责多轮调度与审核。
