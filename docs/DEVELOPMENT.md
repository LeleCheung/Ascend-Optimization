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

当前选题以[最高决策：当前工作范围](../AGENTS.md#最高决策当前工作范围)为准：主线为1015＋1029共30项，早期9项只保留历史材料，不再选择开展新优化、设备复现或方法实验。阶段目标与实验安排在开发文档维护，不写入面向使用者的发布README。

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
| [历史验证记录](archive/validation-20261104.md) | 工程拆分时的初始验证及其范围，不替代当前索引 |

历史恢复标签为 archive/pre-release-20261103，包含旧 project 目录及暂停原件；检出命令、检查点和本地归档位置见 [恢复说明](recovery.md)。恢复历史源码后先核对暂停状态和实际进程，不直接执行旧 launch/prepare 脚本。

## 当前状态与后续维护

截至2026-11-10，独立核心、项目skill及可选后端已有安装和离线验证，固定候选全量回归及SELU／log10生成实验已完成。原合同及语义口径达标26/30，以[当前索引](../benchmarks/current.json)为准；`benchmarks/index.json`是拆分前冻结索引。旧控制器保持停用，新实验不通过历史launch／prepare脚本恢复。

当前阶段收敛为工程整理与三端同步、解耦入口复现，以及30项内的热点区域／等待与空闲分析验证，暂停新增交付数量目标。复现优先复用已有候选和合同，不默认让模型重新生成全部26项；冻结重生成选择少量代表验证方法。`native_group_norm`已相对CANN约1.004×并达标，不仅凭该整体分数安排进一步优化；先从已有逐case证据筛选具体设备性能短板。

| 验证对象 | 当前结论及详细记录 |
| --- | --- |
| 固定候选全量回归 | 独立CLI／KGS回归及后续修复见[回归报告](regression-20261109.md)，不等同于从起点重新生成 |
| SELU新生成 | 0.887010× CANN，正式36/36、双方各42/42、专项33/33；见[恢复报告](selu-skill-recovery-20261109.md)。尚无同规格冻结重生成结论 |
| log10方法迁移 | 开发生成0.851055×、冻结重生成0.846792× CANN；两版正式36/36、双方各66/66、附加15/15；见[迁移报告](log10-method-transfer-20261110.md) |
| 历史候选与新生成的关系 | 保留原三类版本并区分既有源码回归和新生成，详见[版本记录](version-lineage-20261109.md) |

生成实验仍由Codex承担外层取证、只读审查和调度，不能表述为全流程自主运行。既有源码性能、新生成性能及方法迁移效果分别报告；全部候选修改由CC／DeepSeek完成，诊断与方法贡献按[最高决策](../AGENTS.md#最高决策诊断与候选实现分工)记录。

后续工程工作需要明确以下接口：

- 根据选定模型宿主，将取证、上下文、候选生成、评测和反馈串接为可恢复的多轮调度。
- 建立外部已验证经验进入包内知识的维护流程，保持分类、去重和来源可追溯。
- 网页前端读取统一 JSON，并区分事实、假设、正式评测与未完成采集；当前仓库提供结果合同，尚未实现前端。

这些是后续开发方向，不表示恢复旧控制器。当前远端保持私有；公开发布或Release单独安排，保留第三方版权、许可证及真实来源。

## 服务器集成验证入口

核心分析入口保持`python -m hgkda`，skill入口保持`skills/hgkda/SKILL.md`。服务器集成验证从仓库根目录执行`python -m tools.skill_validation --help`，按动作准备实验、调用模型、采集或评测；参数和模块分工见[验证工具说明](../tools/skill_validation/README.md)。这是显式单项调度，不是完整无人值守外循环。

通用工具留在`tools/skill_validation/`，专项检查和参考构建集中到[tools/operators/](../tools/operators/README.md)，旧SELU上下文兼容位于`skill_validation/legacy/`。直接改写候选的旧实验控制器退出当前入口，原字节保留在[tools/archive/](../tools/archive/README.md)。早期`prepare_and_run.py`与固定输入的`probe_host.py`保留在Git历史。新实验使用`prepare`及`operator_transfer`配置；不能将专项检查名称误解为适用于全部算子。

`benchmarks/`中的提示词、方法副本、模型响应和日志是冻结证据，不在其中开发工具或修改历史快照。`build/`、`*.egg-info/`与`__pycache__/`为可再生产物，正常运行不依赖它们。工作环境及临时控制资料位于仓库外，具体边界见[目录说明](directory-layout.md)。

历史版本定义、三类成绩和发布回执统一维护于[版本记录](version-lineage-20261109.md)，开发指引不重复保存逐轮轨迹。下一阶段方法迁移的设计及已完成试点见[实验计划](workflow-learning-plan-20261110.md)。
