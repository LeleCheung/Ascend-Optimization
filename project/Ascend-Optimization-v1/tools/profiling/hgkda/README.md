# HG-KDA 分阶段体检与 KG 接入

2026-10-31完成[最后两项交付](../../../operators/campaigns/remaining-final-20261031/交付报告.md)：`fix` 0.862×、`expm1_` 0.879×，累计24/30常规稳定达标。[remaining_final/](remaining_final/README.md)串联Ascend C设备与C++宿主变体、固定一次确认、原生路径及冻结新形状检查。本轮新增[执行路径诊断脚本](skills/hgkda/scripts/analyze_execution_paths.py)：核验源码/请求/合同，按逐case对数贡献排序，分离候选与参考耗时变化，关联已核验trace，并给出宿主或设备方向的待验证动作。它可离线复用；复杂机制与实现选择仍由Codex完成，本轮未新增CC/DeepSeek调用、RTL或完整关键路径重建。已完成14份正式请求、12条设备路径、2份真实Triton IR及5项离线测试核验。

2026-10-30完成[剩余九项攻关](../../../operators/campaigns/remaining-nine-20261030/交付报告.md)，入口为[remaining/](remaining/README.md)：复用固定KGS合同，新增来源可见Ascend C＋Triton布局回退、对齐原生路径语义门禁、临界项确认和结构化交付。六项新增稳定达标，fix第三次低于0.8×、expm1_性能和求逆逐值合同问题单列；52条设备路径、9份实际IR及37份正式请求已核验。分类知识保留双缓冲选择、启动与设备计时分离、特殊值必须经过快速路径等经验；Codex实现与CC/DeepSeek搜索分别记录。

2026-10-27完成[单算子三步演示](../../../operators/campaigns/agent-demo-20261027/三步演示-核验版.md)，统一入口见[agent_demo/](agent_demo/README.md)：同输入成对采集→Agent选择证据工具→源码／IR／Roofline及热点诊断→有限动作→完整KGS→新采集与预测复核。CC＋DeepSeek真实执行三次角色调用，模板和控制器由Codex实现。共享B候选本批实际1.023×起点，相对PyTorch0.884×，两版42/42及18＋24检查通过；复现历史同字节成功源码。起点1087条、候选926条模拟器指令，部分证据与子进程退出异常单列；核验版报告更正模型过度归因，原回执保留。

2026-10-27完成[矩阵乘复用与流水优化](../../../operators/campaigns/matmul-pipeline-20261027/精简汇报.md)：新增[matmul_pipeline/](matmul_pipeline/README.md)，把Roofline／IR证据转成有限多输出块变换，并由CC＋DeepSeek实际选择追加流水方案。最终共享B候选较同批父版实际设备加速1.040×，相对PyTorch0.895×，42/42及18＋24补充检查通过；同模板复用方向四项确认1.059×。硬件总流量基本不变，不能归为减少HBM搬运。成功与失败知识已分类保存；本轮由Codex实现通用模板、编排和核验，模型调用及输入快照独立留证。

2026-10-27完成[数值Roofline与矩阵乘闭环](../../../operators/campaigns/roofline-20261027/最终报告.md)：新增[roofline/](roofline/README.md)，复用导师分析器并校验固定910B计数、GM／L2范围与同精度经验参考；12项同输入数值采集、中文图表和可注入KG角色的摘要已交付，CC＋DeepSeek实际消费留证。完整42/42＋18/18通过，但单输出tile候选相对父版0.888×，保留父版，无新增性能收益。官方GM0.8TB/s模型与本机观测不匹配，物理上界保持未决；经验线不称峰值利用率。旧批次“不具备数值Roofline”的记录按原时间保留。

2026-10-26完成[设备机制系统闭环](../../../operators/设备机制系统闭环-20261026.md)：归约原合同27/27，机制组设备计时相对PyTorch1.509×、相对本批父版1.083×；独立新形状完整调用1.113×、24/24。矩阵乘未获设备改善，保留父版。新增[device_mechanism/](device_mechanism/README.md)，连续执行设备干预、分层补测、同预算模型配置、完整评测、冻结后检验及自动报告。方法/评测213项、附件/CSV976项核验通过；模拟器失败与RTL预留单列，原合同计时错误标签以源码快照更正。本轮没有证明结构化诊断在独立集上优于原始证据组。

2026-10-26新增[机制闭环原型](../../../operators/机制闭环原型-20261026.md)：同编译产物调用干预→竞争假设→模型选动作→通用模板变换→完整评测→预测核验。开发SELU和冻结后独立asinh分别相对本批FlagGems＋共享适配为1.457×、1.448×，均51/51＋21/21；307项证据核验通过。规则、探针、检查器及变换器在独立检验时保持相同SHA，未加入asinh专用数学改写。模块与复现入口见[mechanism/](mechanism/README.md)；早期重复执行、上游布局错误和tanh暖态JIT不可捕获均保留。本模块是调用机制原型，未实现设备内部依赖重建；不改写历史成果归属。

2026-10-26新增[剩余三项验收与参考路径诊断](../../../operators/campaigns/acceptance-three-20261026/README.md)：池化连续输出分块较同批父版完整调用实际加速1.547×；共轭既有源码完成独立NPU分解对照；累计积分默认参考16次出现4处原容差区间无交集，V2接口仍不稳定。新增[参考可信性与NPU分解](skills/hgkda/references/性能指标/参考可信性与NPU分解.md)，按叶子注册、trace、稳定性、分解及源码／IR组织诊断。补充参考不替换官方原参考，本批由Codex实现和编排，未新增CC搜索。96项证据核验通过，正式复现入口为`reproduce-acceptance.py`；扫描反例入口为`reproduce-scan-reference.py`。

[夜间连续大goal](../../../operators/夜间连续优化交付-20261023.md)已取得完整结果：GELU 2.271×、标准Philox修复uniform 1.325×、rad2deg_ 1.117×、SELU 0.837×；scatter_add 0.419×未达标，累计积分未通过全部数值检查。`run-endurance.py`与独立补充队列在宿主tmux串行执行，自动衔接有限搜索、门禁、一次正式完整KGS、采集与证据审计。

新增[uniform整数机制知识](research/uniform无符号整数修复-20261023.md)：通过确定性设备反例区分乘法、逻辑右移与参数状态；显式有符号移位加掩码、位运算counter进位绕过固定后端的错误与不支持转换。最终标准Philox逐位16项、语义18项、随机状态3项和完整KGS18项通过。原错误高分、失败修复与最终版分别归档，CC生成与Codex机制修复分别登记。

主批次最终审计还会读取所有已通过门禁的起点与搜索候选，核对它们实际上传的合同、源码和逐 case 正式延迟，独立确认入选者耗时最低。相对 PyTorch 分数更高、用例集合不同、非有限延迟或同源码出现冲突正式延迟，均不能绕过选版核验。追加采集修复批次按其登记的兼容修复目标独立核验。

随机数优化另有 `check-uniform-philox.py`：用独立 Python 整数参考逐位核验手写 Philox，区分输出排列错误、整数公式错误和参数传递／设备实现差异。`finish-uniform-philox-repair.py` 在主队列后取得同一卡锁，必要时调用 `prepare-uniform-philox-repair.py` 生成独立修复副本，再执行原语义门禁和一次完整 KGS。搜索内均值方差与随机状态检查通过，不能代替这项算法核验；Codex 追加修复与 CC 生成分别登记。

`stage-endurance-check.py` 将当前候选、登记身份一致的补充检查器及冻结 Native 合同复制到 KG 工作区的新 `tmp/` 目录，输出可供原生 `submit_debug_job` 调用的请求。模型可以在搜索内读取语义失败、修复后再评测，避免整批结束才发现高分候选不合格。打包本身不使用 NPU；调试检查由当前串行 KG/KGS 执行，不另开卡上任务。只有源码 SHA 相同才能复用检查结果；搜索内检查不代替最终补充门禁和一次完整正式评测。该入口覆盖 rad2deg_、cumulative_trapezoid、scatter_add、uniform、gelu_and_mul；SELU 保留三项专用检查。实际采用情况以调试请求和回执为准，工具存在不代表模型已经调用。

打包清单登记打包器本身的 SHA，并拒绝调试接口文本传输会改写换行的文件。遇到 CRLF/CR 源码时，应先生成并登记换行规范化后的新候选，再打包；工具不会偷偷改写源码或冻结合同。

用于定位合同的 `plan.json` 元数据会显式统一换行；打包清单同时保存原 plan 和打包 plan 的 SHA，并核对 JSON 内容完全相同。四份冻结合同及候选源码仍要求原字节通过文本接口传输。

本批[scatter_add 探针复核](../../../operators/campaigns/endurance-20261022/research/scatter_add/second-pass-probe-cpu-review/README.md)保存了实际诊断代码的 CPU 反例：错误的三维 stride 使累加模型大面积偏离参考。单行修复后九组 CPU 复核通过，原 NPU 探针结果保留；设备错误后的异常耗时不作为有效优化证据。

相关约束已写入[诊断探针可信性](../../../operators/campaigns/endurance-20261022/knowledge/诊断探针可信性.md)，供后续新覆盖层读取：调试任务成功不等于每个变体正确，索引模型先做 CPU 校验，原子更新检查初始化，设备错误后的耗时需独立验证。已冻结的 scatter 角色不回改。

本目录对应 2026-10-15 会议后的下一阶段。固定版本见[基线锁定](../../../../../manifests/hgkda-stage2.lock.json)，15 个对象的登记见 [targets.json](targets.json) 和[下一阶段计划](../../../operators/下一阶段计划-20261015.md)。

最新[缓冲与搬运批次](../../../operators/campaigns/buffer-pointwise-20261022/README.md)已完成：log10_ 0.875×，较同批父版实际加速1.125×；selu_保留旧源码，本批复测0.898×。36项正式测试、42＋12＋9项补充检查和48项证据核验通过。新增缓冲生命周期知识、数值范围门禁、异常恢复及同字节评测复用；CC生成与Codex整合分别记录。复现使用`reproduce-paired.py --batch buffer-pointwise-20261022 --tag final --output /data/hanle/ascend-optimization/runs/新目录`，模拟器失败与最终真机采集分开报告。

2026-10-22新目标：[log10_、selu_成对诊断与优化](../../../operators/campaigns/paired-pointwise-20261022/README.md)。新增同输入成对采集与设备计数比较，结合实际launcher/IR提出修改。`build-kg-overlay.py --paired-evidence 摘要.json --native-context`可把冻结seed的成对证据注入新KG分析角色；角色明确区分起点证据和新候选测量。新批次最终默认一次完整KGS评测，历史批次保留原验证次数。

该目标本批已完成闭环：log10_最终0.737×、selu_最终0.790×；各36/36、42＋12补充检查通过，KGS原生profiling与四组成对采集成功，48项证据核验通过。成果包含当前地址启动、log10中间精度优化、SELU按dtype分块，仍未达到整组0.8×。`finish-paired.py`串行收尾，`audit-paired.py`离线核验证据，`reproduce-paired.py --output /data/hanle/ascend-optimization/runs/新目录`独立复现交付源码。快路径依赖固定FlagTree的私有launcher ABI，换版本须重新验证。冻结覆盖层未回溯应用后续知识更新。

2026-10-21 正在执行[夜间大goal](../../../operators/夜间大goal-20261021.md)：`prepare-nightly.py` 构建冻结的Native合同，`run-nightly.py` 串行体检并运行原生KG，`finalize-nightly.py` 等待卡锁后补充语义检查与统计核验。`report-nightly.py` 检查实际上传bundle、源码、用例、两次完整复验和profiling采集/分析证据。控制端用 `../../control/nightly-status.py` 查询；仅启动队列或加载角色不代表完成优化。

本批已观察到原生KG主动从metrics升级到CANN模拟器；首次instruction请求超时，成功metrics与失败采集分别保留。配置、预算及后续方法问题见[指令采集观察](research/原生KG指令采集预算-20261021.md)。该记录不改变运行中的冻结覆盖层。

## 方法与实现

导师总工程图的中文转录、Mermaid流程与HG-KDA实际贡献对照见[导师总工程与HG-KDA贡献](../../../../../docs/导师总工程与HG-KDA贡献.md)。HG-KDA目前是可接入KG的skill、知识和脚本模块，采集与评测依赖KGS；不将它等同于整套融合提取、wrapper、JIT与发布平台。

入口 skill 是 [skills/hgkda/SKILL.md](skills/hgkda/SKILL.md)。知识同时按算子类型和性能指标分类，路由见[知识索引](skills/hgkda/references/知识索引.md)。

`skills/hgkda/scripts/diagnose.py` 从逐 case 数据产生瓶颈假设、知识路由与下一阶段采集建议。简单算子先用 metrics；复杂路径或多流水活跃难以解释时，升级 instruction 并检查同步与热点。当前不是定量 component-based Roofline。

`../hgkda_adapter.py` 连接现有控制器与 KGS：每轮最多新增一次代表 case 的指令采集，校验附件，按候选源码中的 jit 符号过滤并提取模拟器热点。原始附件保留，模型获得结构化事实和分类知识。缺少同候选证据、后端不支持或请求结果未知时，不假装完成诊断。

新批次的指令目录使用 case_id 的摘要命名，支持 Windows/Linux 同步。2026-10-15 首批原始证据中的含冒号路径已仅改文件名，内容不变，映射见该批的 `evidence-path-relocations.json`；原始 JSON 与提示词中的历史路径依此映射查找。

## 使用现有控制器

仅在新的批次给 `optimization_campaign.py` 增加 `--hgkda`，继续使用原 KGS 完整评测、最佳保留和两次复验。默认不带此参数时沿用旧流程，历史结果不重新解释。

```bash
python tools/profiling/optimization_campaign.py \
  --operator rad2deg_ --seed operators/rad2deg_/candidates/flaggems-fixed.py \
  --output operators/rad2deg_/reports/hgkda-新批次 \
  --knowledge tools/profiling/hgkda/阶段通用知识.md \
  --rounds 2 --plain-json --provider-thinking disabled --hgkda
```

控制器将全部 skill 文件摘要写入新批次配置，恢复时知识改变会停止。采集选择与模型预算须随结果报告，不能把已有知识指导的结果声称为无先验自动发现。

`rad2deg_` 新候选在正式评测前自动执行21项补充语义检查，并核对源码 SHA。它覆盖当前正式合同之外的原地别名、非连续布局和奇数长度；其他算子尚未接入这项专用门禁。首批真实结果、修复贡献与未采纳指令候选见[首项交付](../../../operators/会议后首项交付-20261015.md)。

## 插入 KG 原生分析流程

`build-kg-overlay.py` 将固定 KG 生成到独立目录，只在两种原生 profiling 分析角色后追加 HG-KDA 方法与分类知识，并附带标准 skill。KG 原有 MCP 工具、证据登记合同、Coder 和 KGS 评测继续使用。原生角色仍由模型推理和调用采集工具；确定性规则脚本的自动执行由上面的控制器适配器负责，两种模式分别记录。

在 910B 宿主机或带符号链接权限的 Linux 环境生成：

```bash
p=/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1
python3 "$p/tools/profiling/hgkda/build-kg-overlay.py" \
  --output /data/hanle/ascend-optimization/runtime/hgkda-新批次/sources

export KG_SOURCE_ROOT=/data/hanle/ascend-optimization/runtime/hgkda-新批次/sources
bash "$p/tools/evaluation/run-kg-operator-910b.sh" \
  rad2deg_ profile hgkda-新批次 2 "$p/operators/rad2deg_/candidates/flaggems-fixed.py"
```

覆盖层记录原角色与扩展角色 SHA，源码根目录通过 `KG_SOURCE_ROOT` 显式选择。简单旧入口可能不自动启用 skill，但两种分析角色都已带上方法说明；不能把目录中存在 SKILL.md 当作模型实际调用的证据。新原生实验需核对工作区实际加载的角色和 MCP 分析登记。

后续新覆盖层可在构建命令中增加 `--native-context`：把该轮冻结的Native输入规格附加到 `get_profile_context` 的 `hgkda_workloads`，并校验候选、测量、逐case参数及profiling集合SHA。原有测量字段保持原值，避免模型在 `axes` 为空时猜错形状。已在真实MoE第二轮工作区完成只读接口验证，详见[MoE分块与计时诊断](research/MoE归约的分块与计时诊断-20261021.md)。本批 `night-20261021-v3` 的冻结覆盖层没有启用这项后续扩展，不把接口验证写成优化收益。

## 离线读取 OpProf 设备计数

`opprof_facts.py` 读取KGS保存的 `msprof-op-report.tar.gz` 和对应 `manifest.json`，验证附件SHA、候选solution SHA、case和kernel身份，输出逐block搬运/流水/冲突指标分布、报告的搬运总量及计数加权L2读命中率。缺失值保持缺失，比例不相加，原始附件不解压或修改。

使用Python 3.8及以上；910B宿主默认Python 3.7，请在 `tle_yy` 内使用 `/usr/local/python3.11.15/bin/python3.11`。此过程离线读取文件，不调用NPU。假设变量 `profile_dir` 指向归档中包含附件与manifest的目录：

```bash
python opprof_facts.py \
  --archive "$profile_dir/msprof-op-report.tar.gz" \
  --manifest "$profile_dir/manifest.json" \
  --expected-archive-sha256 957b7b956505fc4b29578384ade826ec6b85ad039941f32c883506fe71c41432 \
  --expected-solution-sha256 f9fff605fcdd834bae26008521efd331194da3e04f0d179d1caa34ee924d355d \
  --case-id sglang_moe_sum_reduce-00004-timing \
  --kernel _moe_sum_reduce_kernel --output /新目录/moe-opprof-facts.json
```

这些SHA与case参数仅属于本次MoE HG第二轮。其他候选须从各自原始identity/manifest和附件核对，输出路径必须未存在。当前真实样本的设备计数有效，但模拟器阶段超时，因此输出标记 `partial_collection=true`、`instruction_listing_available=false`；失败指令请求和有效设备采集分别报告。

表中的block记录不是物理核数量；逐block带宽usage不是全卡HBM峰值利用率，厂商memory接口计数也不自动等于物理HBM流量。该工具尚未自动接入本轮冻结的KG分析角色。真实结果和后续诊断见[MoE分块与计时诊断](research/MoE归约的分块与计时诊断-20261021.md#opprof逐block设备计数)，专项测试为 `tests/test_opprof_facts.py`。

分阶段诊断脚本现要求指令请求 `status=completed` 且存在 `instruction_listing`，才登记指令证据；只有日志、OpProf设备报告或失败回执时保留metrics阶段。适配器仍不会自动重提已有失败请求。对应回归测试已覆盖这次MoE“设备计数成功、模拟器失败”的情况；本次改动只供后续批次使用。

`selu_` 的真实指令、调用路径和分块探针见[逐元素启动诊断](research/逐元素算子的启动路径与分块诊断-20261021.md)。`audit-selu-launch.py` 是需要同一卡锁的额外NPU诊断，核对固定候选的缓存key、真实直接启动和计算，不作为正式性能成绩；不是离线OpProf解析的一部分。

## 夜间批次的最终源码交付

原主队列和补充语义任务结束后，在容器Python 3.11或控制端Python 3.8及以上运行：

```bash
python tools/profiling/hgkda/export-nightly-deliverables.py \
  --batch-dir operators/campaigns/night-20261021-v3
```

工具重新核验两个重点算子的冻结合同、源码身份、补充语义及两次完整复验；额外修复还核验父源码谱系。仅在这些检查通过后，按相同case的实际延迟选出有效最佳版本，将原始源码字节写入批次 `deliverables/<算子>/<源码SHA>/main.py`，生成 `交付清单.md` 和 `交付清单.json`，并调用现有比较工具输出该源码与固定FlagGems的逐case Markdown、CSV及JSON。尚有待完成候选或重点算子没有有效源码时返回非零。

排名使用固定FlagGems延迟与候选延迟之比，跨两次复验取几何平均，避免PyTorch参考耗时波动改变候选排名。原始相同预算实验和额外Codex修复分别标记；导出交付源码不改变原始四组报告的严格门禁或失败证据。

`run-semantic-repair.py` 还支持单独登记的 `moe_semantic_repair` 模式：保留原始父源码及36项语义失败快照，先通过全部36项、再执行两次原8项正式合同。其核验与原value修复共用 `verify-semantic-repairs.py`，算子、完整case集合、检查器、源码和设备身份均须对应。`moe-stride-repair` 和 `moe-hg-layout-repair` 属于额外Codex修复，不在旧postflight预置等待名单中，收尾时必须分别检查；交付导出工具已识别这两个批次。

## 固定基线与验证

```bash
python tools/profiling/hgkda/verify-baseline.py \
  --server http://127.0.0.1:19655 --output /新批次/baseline-verified.json
```

核对固定 FlagGems 的逐文件 SHA 与 KGS 报告的 FlagTree、Triton、CANN、驱动版本。PR #1238 的编译器变更另设实验组，不能在同一组对照期间升级。

离线测试见 `tests/test_diagnosis.py`，验证缺失指标、非法比例、重叠流水升级、服务能力限制及真实历史指令的候选热点筛选。完整正确性与性能必须由 910B 的 KGS 验证。
