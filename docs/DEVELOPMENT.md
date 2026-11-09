# HG-KDA 开发指引

本文面向维护者及接续开发的 Agent，介绍实际代码、运行边界、验证方式和当前工程状态。项目用途、安装和快速使用见 [发布 README](../README.md)，开发约束见 [AGENTS.md](../AGENTS.md)。

## 文档分工

| 文档 | 维护内容 |
| --- | --- |
| README.md | 面向使用者的稳定入口：项目能力、方法、安装、示例与适用范围 |
| docs/DEVELOPMENT.md | 面向维护者的工程入口：代码组织、调用关系、开发验证、状态与待完善事项 |
| AGENTS.md | 必须遵守的实现、证据与共享环境约束 |
| benchmarks/current.json | 最新状态及审计原件位置，独立数学参考合同单列 |
| benchmarks/index.json 及历史原件 | 拆分前的源码、合同和性能身份，冻结保留 |
| docs/ 下的专题文档 | 后端配置、机器目录、验证记录和历史恢复细节 |

修改接口时同步更新使用示例；修改实验状态时先更新绑定源码与合同的结果索引。安装验证、离线复算和新算子达标分别记录。

## 核心代码与调用关系

实际主体位于 [src/hgkda/](../src/hgkda/)，skill 位于 [skills/hgkda/](../skills/hgkda/)。核心安装配置见 [pyproject.toml](../pyproject.toml)。

当前入口的调用关系见[HTML 流程图](hgkda-architecture.html)，可直接用浏览器打开并打印为 PDF。

    模型宿主读取 SKILL.md
        → 调用 run_hgkda.py / hgkda / python -m hgkda
        → cli.py 分派至分析模块或后端适配器
        → 返回结构化证据与知识
        → 模型判断机制、生成修改
        → 审查候选，调用 KGS 验证并记录反馈

| 模块 | 职责 |
| --- | --- |
| [cli.py](../src/hgkda/cli.py) | 统一参数解析、命令分派和结果输出 |
| [artifacts.py](../src/hgkda/artifacts.py) | 输入文件身份、JSON 封装及不覆盖式输出 |
| [compare.py](../src/hgkda/compare.py) | 校验成对采集，统计指标分布、总和、比值和平均搬运粒度 |
| [diagnostics.py](../src/hgkda/diagnostics.py) | 按算子类型与流水比例生成规则假设及补测建议 |
| [hotspots.py](../src/hgkda/hotspots.py)、[regions.py](../src/hgkda/regions.py) | 筛选候选内核指令，聚合源码区域；有时间轴时计算区间活动、等待和空闲 |
| [roofline.py](../src/hgkda/roofline.py) | 根据 GEMM 工作量、同次采集及经验参考计算数值模型 |
| [context.py](../src/hgkda/context.py) | 核对源码身份，将对照、已提供的实际 IR、Roofline、热点及知识组织为模型上下文 |
| [feedback.py](../src/hgkda/feedback.py) | 核对完整性能与case清单，按几何平均损失和输入规模选择后续诊断对象，返回继续优化或达标决策 |
| [data/knowledge/](../src/hgkda/data/knowledge/) | 方法约束、算子类别、指标规则与历史机制经验，随 Python 包安装 |
| [curation.py](../src/hgkda/curation.py) | 知识证据核验、同作用域去重、历史保留与限长检索 |
| [gate.py](../src/hgkda/gate.py)、[validation.py](../src/hgkda/validation.py) | 静态筛查、源码审查绑定与算子补充检查结果判定 |
| [adapters/](../src/hgkda/adapters/) | KGS HTTP 评测、内容寻址合同包及 msprof 显式命令采集 |

建议从 cli.py 了解入口，再阅读 context.py 理解证据组织，最后查看对应分析模块与知识文件。

## skill、模型与后端的边界

| 部分 | 当前职责 |
| --- | --- |
| skill | 指导宿主模型选择取证深度、形成可验证修改并调用工具 |
| Python 核心 | 确定性计算、身份核验和上下文组织，离线运行不导入 KG/KGS |
| 模型宿主／可选 KG | 模型调用、机制解释、代码生成和多轮优化编排 |
| KGS | 使用固定测试合同完成正确性与性能评测 |
| msprof | 真机或模拟器采集；报告解析和证据校验另行执行 |

context 接受已经采集并绑定源码的 IR、指令清单和数值参考；它本身不运行编译器、采集器或模型。当前与 KG 的集成边界是上下文 JSON 和工具调用，具体自动衔接由宿主负责。

模型与后续网页共用 [hgkda.result/v1](../schemas/result.schema.json) 结果封装。context 将 facts、hypotheses、knowledge 和 agent_task 分开，便于区分采集事实、规则假设和模型任务。

curate/retrieve 对外部经验记录执行确定性核验与检索；context 默认加载包内分类知识。将新经验纳入包内知识须经审查，并记录来源和适用条件，不把模型生成的建议直接标为成功规律。

## 开发环境与验证

在独立 Python 3.10+ 环境中，从仓库根目录安装开发版本并验证：

    git lfs pull
    python -m pip install -e .
    python -m hgkda doctor
    python skills/hgkda/scripts/run_hgkda.py doctor
    python -m unittest discover -s tests -v

离线调用使用 [README 示例](../README.md#快速示例)，输出放在仓库外的新目录。当前测试覆盖身份错配、缺失与零分母、时间重叠、合同包、候选审查、请求异常和知识整理。

改动分析模块时检查对应测试和真实离线案例；改动 KGS/msprof 适配器时，在核对环境与设备分配后执行必要的后端验证。运行时、venv、编译缓存、日志和模型设置放在仓库外。

Windows 与 910B 的实际目录及已安装环境见 [目录与运行环境](directory-layout.md)，服务参数和协议边界见 [后端说明](backends.md)。共享环境和严格 Triton 的约束统一遵循 [AGENTS.md](../AGENTS.md)。

## 实验、来源与历史

| 位置 | 内容 |
| --- | --- |
| [benchmarks/current.json](../benchmarks/current.json) | 当前算子分组、顺序、验收状态及证据索引 |
| [benchmarks/index.json](../benchmarks/index.json) | 拆分前的固定候选及合同来源 |
| [benchmarks/operators/](../benchmarks/operators/) | 各算子的冻结源码、审查与核心结果 |
| [examples/](../examples/) | 可运行的精选证据及 KGS 请求 |
| [manifests/](../manifests/) | 依赖版本、迁移来源、文件身份与核验记录 |
| [sources/](../sources/)、[third_party/](../third_party/) | 可选 KG/KGS 与固定 FlagGems 快照，不参与核心包安装 |
| [验证记录](validation.md) | 工程整理后的实际执行结果及其范围 |

历史恢复标签为 archive/pre-release-20261103，包含旧 project 目录及暂停原件；检出命令、检查点和本地归档位置见 [恢复说明](recovery.md)。恢复历史源码后先核对暂停状态和实际进程，不直接执行旧 launch/prepare 脚本。

## 当前状态与后续维护

截至 2026-11-04，独立核心、项目 skill、可选后端及目录整理已完成；Windows 和 910B 的安装、离线示例与一次既有严格 Triton 算子的 KGS 完整评测已有记录。旧算子优化仍暂停，验收数量和各算子状态以 [实验索引](../benchmarks/index.json)为准。

2026-11-09 完成 [25 项固定候选回归](regression-20261109.md)：经独立 CLI 调用 KGS，776/776 正式用例及 2366/2366 补充检查通过，性能为 23/25 达标。修复候选、Uniform 新参考和累计积分独立数学参考合同在新批次分别记录，当前状态见 current.json。该批次与旧优化控制器分别管理，运行结果不覆盖历史原件。

后续工程工作需要明确以下接口：

- 根据选定模型宿主，将取证、上下文、候选生成、评测和反馈串接为可恢复的多轮调度。
- 建立外部已验证经验进入包内知识的维护流程，保持分类、去重和来源可追溯。
- 网页前端读取统一 JSON，并区分事实、假设、正式评测与未完成采集；当前仓库提供结果合同，尚未实现前端。

这些是后续开发方向，不表示恢复旧优化任务。当前远端保持私有；公开发布或 Release 单独安排，保留第三方版权、许可证及真实来源。

## 独立优化闭环与三类版本记录

2026-11-09 本轮仅选择具有合格 CANN 参考的 `selu_` 进行实际生成验证，其余算子复用已完成的回归证据。目标是验证宿主 Agent 加载独立 HG-KDA skill 后调用工具完成诊断、修改和评测，而不仅是通过新 CLI 复测已有源码。具体边界、三类版本及复核方法见 [版本记录](version-lineage-20261109.md)。

流程为：固定 FlagGems／FlagTree 版本、CANN 参考与测试合同 → 宿主 Agent 读取 skill → 成对采集与指标比较 → 按需读取源码、实际 IR、Roofline 与热点证据 → 生成严格 Triton 候选 → 路径审查、完整 KGS 评测及补充检查 → 保留有效候选与更新经验。复用现有工具，按实际执行中发现的缺口补齐调度；模型、采集和每次修改均保留轨迹。

从固定的起始实现开展优化，不直接把历史最佳源码作为生成成果。允许使用已有分类知识，但须记录输入的知识版本及人工干预。已经完成的固定候选回归和本轮 Codex 分析后修改的版本分别保留，不能表述为独立 skill 从起点自动生成的结果。

启动路径与性能反馈的恢复方法见 [SELU独立skill性能恢复](selu-skill-recovery-20261109.md)。包内经验检索现在显式返回精确／类型迁移数量；性能失败通过 `feedback` 回传。服务器工具的模型上下文须绑定当前源码，不能复用旧SHA的采集；独立启动诊断与完整KGS计时分别保存。

本轮新生成版已由0.210037×提升至0.887010× CANN，正式36/36、双方各42/42和专项33/33通过。历史独立回归1.039292×继续作为交付；总达标数仍为26/30。新生成候选及全部中间失败单独保存，包含明确的Codex方法指导。

本轮完成证据核验及 GitHub 同步后，更新语雀 [FlagGems Table](https://bosc.yuque.com/up3wk0/twwy86/vcqrbcy2hp7gds4g)。保留原算子范围、来源分组与顺序，表格仍以表格为主体，不加入实验过程长文。三类记录如下：

| 类别 | 保留内容 |
| --- | --- |
| 历史最佳（不符合当前约束） | 当时的源码与成绩、参考及计时口径，以及不符合当前要求的具体原因；不计入严格 Triton 交付 |
| 历史最佳（严格 Triton） | 经完整正确性、补充检查及实现路径审查的历史候选，保留源码 SHA 和原始评测 |
| 独立入口验证版（严格 Triton） | 经独立入口完整验证的候选及本次成绩；标明既有源码回归或 Agent 新生成／优化，保留日期及证据 |

三类记录可能共享同一份源码，但不同测量分别保存。仅在同参考、同合同和同计时口径内选取最佳候选，不以同一源码反复计时的最高分替代正式统计。新数学参考合同单列；缺少历史证据时标记缺失，不补造数字。更新语雀前读取最新正文，保留已有固定版本说明及用户编辑。
